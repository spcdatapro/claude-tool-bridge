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
_WS_RE = re.compile(r"\s*")
_ONLY_WRAPPERS_RE = re.compile(r"(?:\s|</?(?:antml:)?function_calls\s*>)*")
_NS = "antml" + ":"  # optional namespace prefix on the tags (built apart so it survives copy and paste)
_PARTIAL_TEMPLATES = ("</invoke>", "<parameter ", "</" + _NS + "invoke>", "<" + _NS + "parameter ")
_PARTIAL_MAX = 256  # a tag still being streamed is never longer than this
_PARAM_PREFIX_RE = re.compile(r"<(?:antml:)?parameter\s+[^>]*")
_EXAMPLE_RE = re.compile(
    r"(Arguments example|Call ([A-Za-z0-9_\-:.]+) with arguments):\n```json\n(.*?)\n```", re.S
)


class _Incomplete:
    def __repr__(self) -> str:
        return "INCOMPLETE"


INCOMPLETE = _Incomplete()


class _Malformed:
    """The body of the first candidate is not a list of parameters; pos is where it went wrong."""

    __slots__ = ("pos",)

    def __init__(self, pos: int) -> None:
        self.pos = pos


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
    if "```" not in text and "~~~" not in text:
        return []  # no line can open a fence
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


def _at_line_start(text: str, pos: int) -> bool:
    """True if only whitespace and <function_calls> wrapper tags precede pos on its line."""
    line_start = text.rfind("\n", 0, pos) + 1
    return _ONLY_WRAPPERS_RE.fullmatch(text, line_start, pos) is not None


def _is_partial_at(text: str, pos: int) -> bool:
    """True if text[pos:] is a tag still being streamed (cut before its closing '>')."""
    if len(text) - pos > _PARTIAL_MAX or text[pos] != "<":
        return False
    rest = text[pos:]
    if ">" in rest:
        return False
    return any(t.startswith(rest) for t in _PARTIAL_TEMPLATES) or bool(_PARAM_PREFIX_RE.fullmatch(rest))


def _parse_body(text: str, pos: int) -> tuple[list[tuple[str, str]], int] | _Incomplete | _Malformed:
    params: list[tuple[str, str]] = []
    size = len(text)
    while True:
        pos = _WS_RE.match(text, pos).end()
        if pos >= size:
            return INCOMPLETE
        close = _CLOSE_INVOKE_RE.match(text, pos)
        if close:
            return params, close.end()
        opened = _PARAM_OPEN_RE.match(text, pos)
        if opened is None:
            return INCOMPLETE if _is_partial_at(text, pos) else _Malformed(pos)
        value_start = opened.end()
        search = value_start
        while True:
            candidate = _PARAM_CLOSE_RE.search(text, search)
            if candidate is None:
                return INCOMPLETE
            rest = _WS_RE.match(text, candidate.end()).end()
            if _CLOSE_INVOKE_RE.match(text, rest) or _PARAM_OPEN_RE.match(text, rest):
                break
            if rest >= size or _is_partial_at(text, rest):
                return INCOMPLETE
            search = candidate.end()  # a literal </parameter> inside the value
        params.append((opened.group(1).strip(), text[value_start:candidate.start()]))
        pos = candidate.end()


def _scan(text: str) -> tuple[str, Any]:
    """Look at the first <invoke> that starts a line outside code fences.

    Returns (status, value): ("none", None) when there is no candidate yet, ("incomplete", None) while
    it is still being written, ("malformed", position) when its body is not a list of parameters, and
    ("found", Invocation) once it is complete.
    """
    if not text or "invoke" not in text:
        return "none", None
    fences: list[tuple[int, int]] | None = None
    for match in _OPEN_RE.finditer(text):
        start = match.start()
        if not _at_line_start(text, start):
            continue  # mid-line: a quoted example, not a call
        if fences is None:
            fences = _fenced_ranges(text)
        if any(a <= start < b for a, b in fences):
            continue
        if match.group(2) == "/>":
            params, end = [], match.end()
        else:
            parsed = _parse_body(text, match.end())
            if parsed is INCOMPLETE:
                return "incomplete", None
            if isinstance(parsed, _Malformed):
                return "malformed", parsed.pos
            params, end = parsed
        prose = _WRAPPER_RE.sub("", text[:start]).strip()
        return "found", Invocation(match.group(1).strip(), params, start, end, prose)
    return "none", None


def find_first_invocation(text: str) -> Invocation | _Incomplete | None:
    """First <invoke> that starts a line outside code fences.

    Returns INCOMPLETE while the first candidate is still being written, None if there is
    no candidate or it is malformed.
    """
    status, value = _scan(text)
    if status == "found":
        return value
    return INCOMPLETE if status == "incomplete" else None


# Characters a StreamWatch may examine for one reply before it stops looking (a safety net: the
# normal cost is a few parses of the text so far; this bounds replies built to make it parse a lot).
PARSE_BUDGET = 4_000_000
_WATCH_OVERLAP = 192  # a tag split across chunks is still seen whole


class StreamWatch:
    """Finds the first complete <invoke> in a stream without parsing every chunk again.

    A parse is only worth its cost when something that can change its answer has just arrived: an
    opening tag at the start of a line while no candidate is known, then a closing </invoke> once one is
    being written. Everything else (parameter values, prose, code) costs one short regex over the new text.
    """

    __slots__ = ("budget", "pending", "done")

    def __init__(self, budget: int = PARSE_BUDGET) -> None:
        self.budget = budget
        self.pending = False  # a candidate is open and waits for its </invoke>
        self.done = False

    def feed(self, chunk: str, full: str) -> Invocation | None:
        """Call with every streamed chunk; returns the Invocation once, when it completes."""
        if self.done:
            return None
        fresh = len(full) - len(chunk)  # where the new text begins
        pattern = _CLOSE_INVOKE_RE if self.pending else _OPEN_RE
        for match in pattern.finditer(full, max(0, fresh - _WATCH_OVERLAP)):
            if match.end() > fresh and (self.pending or _at_line_start(full, match.start())):
                break
        else:
            return None
        self.budget -= len(full)
        if self.budget < 0:
            self.done = True
            return None
        status, value = _scan(full)
        if status == "found":
            self.done = True
            return value
        if status == "incomplete":
            self.pending = True
        elif status == "malformed" and len(full) - value > _WATCH_OVERLAP:
            self.done = True  # broken well before the end of the text: more text cannot fix it
        return None


# ---------------------------------------------------------------- server-side stop

# The model is asked to stop inside "</parameter>\n</invoke>", so the text that comes back ends with a
# half tag that never occurs by accident. complete_stopped() recognises it and closes it again.
_STOP_CUTS = (
    ("</paramete", "r>\n</invoke>"),
    ("</" + _NS + "paramete", "r>\n</" + _NS + "invoke>"),
)
STOP_SEQUENCES = [tail for _, tail in _STOP_CUTS]


def complete_stopped(text: str) -> str | None:
    """Put back the end of a reply the model was stopped in, or None if it does not end that way."""
    for mark, tail in _STOP_CUTS:
        if text.endswith(mark):
            return text + tail
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
