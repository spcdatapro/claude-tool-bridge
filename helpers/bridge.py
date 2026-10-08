"""Translate Claude-native <invoke> tool calls into Agent Zero JSON tool requests.

Pure logic, standard library only, no imports from Agent Zero or from other modules of this plugin,
so it can be loaded by path (tests, offline evaluation) without installing the plugin.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any

_OPEN_RE = re.compile(r'<(?:antml:)?invoke\s+name="([^"]+)"\s*(/>|>)')
_CLOSE_INVOKE_RE = re.compile(r"</(?:antml:)?invoke\s*>")
_PARAM_OPEN_RE = re.compile(r'<(?:antml:)?parameter\s+name="([^"]+)"\s*>')
_PARAM_CLOSE_RE = re.compile(r"</(?:antml:)?parameter\s*>")
_WRAPPER_RE = re.compile(r"</?(?:antml:)?function_calls\s*>")
_FENCE_RE = re.compile(r"(```+|~~~+)")
_PARTIAL_TEMPLATES = ("</invoke>", "</invoke>", "<parameter ", "<parameter ")
_PARAM_PREFIX_RE = re.compile(r"<(?:antml:)?parameter\s+[^>]*")
_EXAMPLE_RE = re.compile(
    r"(Arguments example|Call ([A-Za-z0-9_\-:.]+) with arguments):\n```json\n(.*?)\n```", re.S
)


class _Incomplete:
    def __repr__(self) -> str:
        return "INCOMPLETE"


INCOMPLETE = _Incomplete()


@dataclass
class Invocation:
    name: str
    params: list[tuple[str, str]]
    start: int
    end: int
    prose: str


@dataclass
class ToolSpec:
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    example_types: dict[str, str] = field(default_factory=dict)


class Registry:
    """Tools known to the agent, keyed by Agent Zero name, with a native-name lookup."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self.native: dict[str, str] = {}

    def resolve(self, name: str) -> tuple[str, ToolSpec] | None:
        """Return (name to put in tool_name, spec), or None if the tool is unknown."""
        name = (name or "").strip()
        if name in self.tools:
            return name, self.tools[name]
        if ":" in name:
            base, method = name.split(":", 1)
            if base and method and base in self.tools:
                return name, self.tools[base]
        original = self.native.get(name)
        if original and original in self.tools:
            return original, self.tools[original]
        return None


# ---------------------------------------------------------------- parsing

def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    opened: int | None = None
    marker = ""
    pos = 0
    for line in text.splitlines(keepends=True):
        match = _FENCE_RE.match(line.lstrip())
        if match:
            if opened is None:
                opened, marker = pos, match.group(1)[0]
            elif match.group(1)[0] == marker:
                ranges.append((opened, pos + len(line)))
                opened = None
        pos += len(line)
    if opened is not None:
        ranges.append((opened, len(text)))
    return ranges


def _is_partial(rest: str) -> bool:
    """True if rest is a tag still being streamed (cut before its closing '>')."""
    if not rest.startswith("<") or ">" in rest:
        return False
    return any(t.startswith(rest) for t in _PARTIAL_TEMPLATES) or bool(_PARAM_PREFIX_RE.fullmatch(rest))


def _parse_body(text: str, pos: int) -> tuple[list[tuple[str, str]], int] | _Incomplete | None:
    params: list[tuple[str, str]] = []
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return INCOMPLETE
        close = _CLOSE_INVOKE_RE.match(text, pos)
        if close:
            return params, close.end()
        opened = _PARAM_OPEN_RE.match(text, pos)
        if opened is None:
            return INCOMPLETE if _is_partial(text[pos:]) else None
        value_start = opened.end()
        search = value_start
        found = None
        while True:
            candidate = _PARAM_CLOSE_RE.search(text, search)
            if candidate is None:
                return INCOMPLETE
            rest = text[candidate.end():].lstrip()
            if _CLOSE_INVOKE_RE.match(rest) or _PARAM_OPEN_RE.match(rest):
                found = candidate
                break
            if not rest or _is_partial(rest):
                return INCOMPLETE
            search = candidate.end()  # a literal </parameter> inside the value
        params.append((opened.group(1).strip(), text[value_start:found.start()]))
        pos = found.end()


def find_first_invocation(text: str) -> Invocation | _Incomplete | None:
    """First <invoke> that starts a line outside code fences.

    Returns INCOMPLETE while the first candidate is still being written, None if there is
    no candidate or it is malformed.
    """
    if not text or "invoke" not in text:
        return None
    fences = _fenced_ranges(text)
    for match in _OPEN_RE.finditer(text):
        start = match.start()
        if any(a <= start < b for a, b in fences):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        if _WRAPPER_RE.sub("", text[line_start:start]).strip():
            continue  # mid-line: a quoted example, not a call
        if match.group(2) == "/>":
            params, end = [], match.end()
        else:
            parsed = _parse_body(text, match.end())
            if parsed is INCOMPLETE or parsed is None:
                return parsed
            params, end = parsed
        prose = _WRAPPER_RE.sub("", text[:start]).strip()
        return Invocation(match.group(1).strip(), params, start, end, prose)
    return None


# ---------------------------------------------------------------- registry

def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _example_types(tools: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    found: dict[str, dict[str, set[str]]] = {}
    for tool in tools:
        description = str(tool.get("description") or "")
        for match in _EXAMPLE_RE.finditer(description):
            target = match.group(2) or str(tool.get("name") or "")
            try:
                args = json.loads(match.group(3))
            except ValueError:
                continue
            if not isinstance(args, dict):
                continue
            for key, value in args.items():
                found.setdefault(target, {}).setdefault(key, set()).add(_json_type(value))
    return found


def build_registry(tools: list[dict[str, Any]], name_map: dict[str, str] | None = None) -> Registry:
    """Build a Registry from build_responses_function_tools(agent) output."""
    name_map = name_map or {}
    registry = Registry()
    examples: dict[str, dict[str, set[str]]] = {}
    for original, keys in _example_types(tools).items():
        examples[name_map.get(original, original)] = keys
    for tool in tools:
        native = str(tool.get("name") or "")
        if not native:
            continue
        original = name_map.get(native, native)
        schema = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [k for k in (schema.get("required") or []) if isinstance(k, str)]
        example_types: dict[str, str] = {}
        for key, kinds in examples.get(original, {}).items():
            if len(kinds) == 1:
                example_types[key] = next(iter(kinds))
            elif kinds == {"integer", "number"}:
                example_types[key] = "number"
        registry.tools[original] = ToolSpec(original, dict(properties), required, example_types)
        registry.native[native] = original
    return registry


# ---------------------------------------------------------------- coercion

def _declared_types(schema: Any) -> set[str] | None:
    if not isinstance(schema, dict):
        return None
    types: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            types.add(value)
        elif isinstance(value, list):
            types.update(v for v in value if isinstance(v, str))

    add(schema.get("type"))
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            if isinstance(sub, dict):
                add(sub.get("type"))
    if not types and isinstance(schema.get("enum"), list):
        types.update(_json_type(v) for v in schema["enum"])
    return types or None


def _matches(value: Any, types: set[str]) -> bool:
    for kind in types:
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        if kind == "null" and value is None:
            return True
    return False


def _parse_typed(value: str, types: set[str]) -> Any:
    text = value.strip()
    if text in ("True", "False"):
        text = text.lower()
    try:
        parsed = json.loads(text)
    except ValueError:
        return value
    return parsed if _matches(parsed, types) else value


def _trim(raw: str) -> str:
    """Drop one leading and one trailing line break: the model puts values on their own lines."""
    if raw.startswith("\r\n"):
        raw = raw[2:]
    elif raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\r\n"):
        raw = raw[:-2]
    elif raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def coerce(raw: str, schema: Any = None, example_type: str | None = None) -> Any:
    """Convert an XML parameter value. The declared schema type wins, then the type seen in
    A0's own usage examples; with neither, the text is kept untouched."""
    value = _trim(raw)
    declared = _declared_types(schema)
    if declared is not None:
        return value if "string" in declared else _parse_typed(value, declared)
    if example_type and example_type != "string":
        return _parse_typed(value, {example_type})
    return value


# ---------------------------------------------------------------- translation

def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def request_from_invocation(inv: Invocation, registry: Registry) -> dict[str, Any] | None:
    """Build an A0 tool request from an invocation, or None if it should not be translated."""
    resolved = registry.resolve(inv.name)
    if resolved is None or not inv.params:
        return None
    name, spec = resolved
    raw = dict(inv.params)  # a repeated parameter: the last one wins
    declared = set(spec.properties)

    if "tool_name" in raw and "tool_name" not in declared:
        claimed = _trim(raw.pop("tool_name")).strip()
        if claimed and claimed not in {inv.name, name, name.split(":", 1)[0]}:
            return None

    thoughts = _paragraphs(inv.prose)
    if "thoughts" in raw and "thoughts" not in declared:
        text = _trim(raw.pop("thoughts")).strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = text
        if isinstance(parsed, list):
            thoughts += [str(p).strip() for p in parsed if str(p).strip()]
        elif text:
            thoughts += _paragraphs(str(parsed))
    headline = ""
    if "headline" in raw and "headline" not in declared:
        headline = _trim(raw.pop("headline")).strip()

    args: dict[str, Any] = {}
    if "tool_args" in raw and "tool_args" not in declared:
        try:
            nested = json.loads(_trim(raw.pop("tool_args")))
        except ValueError:
            return None
        if not isinstance(nested, dict):
            return None
        args.update(nested)
    for key, value in raw.items():
        args[key] = coerce(value, spec.properties.get(key), spec.example_types.get(key))

    if not args or any(k not in args for k in spec.required):
        return None
    request: dict[str, Any] = {}
    if thoughts:
        request["thoughts"] = thoughts
    if headline:
        request["headline"] = headline
    request["tool_name"] = name
    request["tool_args"] = args
    return request


def translate(text: str, registry: Registry) -> dict[str, Any] | None:
    """Translate the first invocation in text; no JSON-presence check (the caller does that)."""
    inv = find_first_invocation(text)
    if not isinstance(inv, Invocation):
        return None
    return request_from_invocation(inv, registry)


def model_matches(provider: str, name: str, providers: list[str] | None, patterns: list[str] | None) -> bool:
    """True if the chat model is covered by the configured providers and name patterns."""
    provider, name = (provider or "").strip().lower(), (name or "").strip().lower()
    if providers and provider not in {str(p).strip().lower() for p in providers}:
        return False
    if not patterns:
        return True
    candidates = {name, name.rsplit("/", 1)[-1]}
    return any(fnmatch.fnmatchcase(c, str(p).strip().lower()) for p in patterns for c in candidates)
