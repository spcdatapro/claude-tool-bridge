"""Glue between the pure bridge and Agent Zero, shared by the plugin's extensions."""

from __future__ import annotations

import time
from typing import Any

from helpers import cache, extract_tools, plugins
from helpers.dirty_json import DirtyJson
from usr.plugins.claude_tool_bridge.helpers import bridge

PLUGIN_NAME = "claude_tool_bridge"
REGISTRY_KEY = "claude_tool_bridge.registry"
REGISTRY_CACHED_KEY = "claude_tool_bridge.registry_cached"  # the registry of this turn came from the cache
PLAN_KEY = "claude_tool_bridge.plan"
STATE_KEY = "claude_tool_bridge.state"
EARLY_STOP_KEY = "claude_tool_bridge.early_stop"
STOP_OFF_KEY = "claude_tool_bridge.stop_off"  # in params_persistent: turns left without stop sequences
STOP_PAUSE_TURNS = 3  # turns without stop sequences after one cost a call
RETRY_KEY = "claude_tool_bridge.retries"  # in params_persistent: silent retries since the last usable reply
SILENT_RETRY_LIMIT = 2  # replies thrown away in a row before the core is left to deal with them

DEFAULTS: dict[str, Any] = {
    "providers": ["anthropic"],
    "model_patterns": ["claude-*"],
    "early_stop": True,
    "stop_sequence": True,
    "loop_guard": True,
    "prose_as_response": True,
    "log_translations": True,
}

CONFIG_TTL = 5.0  # seconds a plugin config / model check is reused across turns
REGISTRY_TTL = 120.0  # seconds a tool registry is reused across turns

_remembered: dict[tuple, tuple[float, Any]] = {}


# ---------------------------------------------------------------- small caches

def _temporary(agent: Any) -> dict[str, Any]:
    """The per-turn scratch dict (Agent Zero clears it at the start of every loop iteration)."""
    params = getattr(getattr(agent, "loop_data", None), "params_temporary", None)
    return params if isinstance(params, dict) else {}


def _persistent(agent: Any) -> dict[str, Any]:
    """The scratch dict that lives as long as the agent's current monologue."""
    params = getattr(getattr(agent, "loop_data", None), "params_persistent", None)
    return params if isinstance(params, dict) else {}


def _scope(agent: Any, *extra: Any) -> tuple | None:
    """Cache key for what depends on the agent's profile and project; None if it cannot be told."""
    try:
        return (*cache.determine_cache_key(agent), *extra)
    except Exception:
        return None


def _recall(scope: tuple | None, name: str, ttl: float) -> Any:
    if scope is None:
        return None
    hit = _remembered.get((name, *scope))
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]
    return None


def _store(scope: tuple | None, name: str, value: Any) -> None:
    if scope is None:
        return
    if len(_remembered) > 256:
        _remembered.clear()
    _remembered[(name, *scope)] = (time.monotonic(), value)


def reset_caches() -> None:
    _remembered.clear()


# ---------------------------------------------------------------- per-turn plan

def load_config(agent: Any) -> dict[str, Any]:
    """Plugin config over the defaults (a saved config.json replaces default_config.yaml whole)."""
    scope = _scope(agent)
    config = _recall(scope, "config", CONFIG_TTL)
    if config is None:
        config = dict(DEFAULTS)
        saved = plugins.get_plugin_config(PLUGIN_NAME, agent=agent)
        if isinstance(saved, dict):
            config.update(saved)
        _store(scope, "config", config)
    return dict(config)


def _model_covered(agent: Any, config: dict[str, Any]) -> bool:
    from plugins._model_config.helpers import model_config

    chat = model_config.get_chat_model_config(agent) or {}
    return bridge.model_matches(
        str(chat.get("provider") or ""),
        str(chat.get("name") or ""),
        config.get("providers"),
        config.get("model_patterns"),
    )


def applies(agent: Any, config: dict[str, Any]) -> bool:
    """True if the agent's chat model is covered by the configured providers and patterns."""
    try:
        override = agent.context.get_data("chat_model_override")
        scope = _scope(agent, hash(repr(override)), repr(config.get("providers")), repr(config.get("model_patterns")))
        covered = _recall(scope, "applies", CONFIG_TTL)
        if covered is None:
            covered = _model_covered(agent, config)
            _store(scope, "applies", covered)
        return bool(covered)
    except Exception:
        return False


def turn_plan(agent: Any) -> tuple[dict[str, Any], bool]:
    """(config, applies) for this turn: worked out once and shared by the plugin's extensions."""
    params = _temporary(agent)
    plan = params.get(PLAN_KEY)
    if plan is None:
        config = load_config(agent)
        plan = (config, applies(agent, config))
        params[PLAN_KEY] = plan
    return plan


def turn_state(agent: Any) -> dict[str, Any]:
    """Facts about this turn's request, filled in as it is built (see the stream and stop extensions)."""
    return _temporary(agent).setdefault(STATE_KEY, {})


def turn_switches(agent: Any, config: dict[str, Any]) -> tuple[bool, bool]:
    """(stop, guard) for this turn's request: whether it carries the stop sequence at the end of the first call
    and the loop guards. Called once per turn; both are off while a pause is running."""
    persistent = _persistent(agent)
    paused = persistent.get(STOP_OFF_KEY)
    if isinstance(paused, int) and paused > 0:
        persistent[STOP_OFF_KEY] = paused - 1  # a turn of the pause has passed
        return False, False
    return bool(config.get("stop_sequence")), bool(config.get("loop_guard", True))


def stop_failed(agent: Any) -> None:
    """A stopped reply could not be turned into a call: no stop sequences for the next few turns."""
    _persistent(agent)[STOP_OFF_KEY] = STOP_PAUSE_TURNS


def allow_silent_retry(agent: Any) -> bool:
    """Spend one silent retry (throw a reply away and ask again). Once SILENT_RETRY_LIMIT are spent there are no
    more until a usable reply resets them: a silent retry breaks the core's count of consecutive unusable
    replies, and that count is what stops an agent that cannot recover."""
    params = getattr(getattr(agent, "loop_data", None), "params_persistent", None)
    if not isinstance(params, dict):
        return False
    used = params.get(RETRY_KEY)
    used = used if isinstance(used, int) else 0
    if used >= SILENT_RETRY_LIMIT:
        return False
    params[RETRY_KEY] = used + 1
    return True


def retry_pending(agent: Any) -> bool:
    """True if silent retries have been spent and no usable reply has reset them yet."""
    return bool(_persistent(agent).get(RETRY_KEY))


def retry_reset(agent: Any) -> None:
    _persistent(agent).pop(RETRY_KEY, None)


def hit_output_limit(llm_result: Any, state: dict[str, Any]) -> bool:
    """True if the reply ended because it ran out of output tokens, not because a stop sequence fired."""
    limit = state.get("max_tokens")
    used = (getattr(llm_result, "usage", None) or {}).get("completion_tokens")
    return isinstance(limit, int) and isinstance(used, int) and used >= limit


# ---------------------------------------------------------------- tools and translation

def has_json_tool_request(text: str) -> bool:
    """True if text already holds a JSON tool call that Agent Zero (or _context_doctor) can use."""
    request = extract_tools.json_parse_dirty(text)
    if not isinstance(request, dict):
        return False
    try:
        extract_tools.normalize_tool_request(request)
    except Exception:
        return False
    return True


def _build_registry(agent: Any) -> bridge.Registry:
    from helpers import responses_tools

    tools, name_map = responses_tools.build_responses_function_tools(agent)
    return bridge.build_registry(tools, name_map)


def get_registry(agent: Any) -> bridge.Registry:
    """Tool registry for this turn: the one built recently for this profile and project, else a new one."""
    params = _temporary(agent)
    registry = params.get(REGISTRY_KEY)
    if registry is None:
        scope = _scope(agent)
        registry = _recall(scope, "registry", REGISTRY_TTL)
        if registry is not None:
            params[REGISTRY_CACHED_KEY] = True
        else:
            registry = _build_registry(agent)
            _store(scope, "registry", registry)
        params[REGISTRY_KEY] = registry
    return registry


def refresh_registry(agent: Any) -> bridge.Registry:
    """Build the registry again, ignoring the cache (a tool may have been added since it was built)."""
    params = _temporary(agent)
    registry = _build_registry(agent)
    _store(_scope(agent), "registry", registry)
    params[REGISTRY_KEY] = registry
    params.pop(REGISTRY_CACHED_KEY, None)
    return registry


def _tolerant_loads(text: str) -> Any:
    """Agent Zero's own forgiving JSON parser, for objects a model wrote with the quoting wrong."""
    return DirtyJson.parse_string(text)


def plain_text_answer(agent: Any, text: str) -> dict[str, Any] | None:
    """The `response` call for a reply that is only prose, or None when it should be left to the core.

    The core already does this in Responses mode (agent.py, process_llm_result_tools); in chat mode it would
    reject the reply and the model would send the same answer again as a call."""
    body = text.strip()
    if len(body) < bridge.PROSE_MIN_CHARS or not bridge.plain_text(body) or body.endswith(":"):
        return None  # a short sentence announcing the next call, a call or half of one, or a reply that ends on a colon
    if bridge.looks_degenerate(body):
        return None
    if has_json_tool_request(body) or extract_tools.is_misformatted_tool_request(body):
        return None
    if "response" not in get_registry(agent).tools:
        return None
    return {"tool_name": "response", "tool_args": {"text": body}}


def translate_response(agent: Any, text: str) -> tuple[dict[str, Any], int] | None:
    """(request, end of the first </invoke>) or None. Never translates a reply that already
    carries a JSON call outside the invocation: _context_doctor rescues those."""
    inv = bridge.find_first_invocation(text)
    if not isinstance(inv, bridge.Invocation):
        return None
    if has_json_tool_request(text[: inv.start] + "\n" + text[inv.end:]):
        return None
    registry = get_registry(agent)
    request = bridge.request_from_invocation(inv, registry, _tolerant_loads)
    if request is None and _temporary(agent).get(REGISTRY_CACHED_KEY) and registry.resolve(inv.name) is None:
        request = bridge.request_from_invocation(inv, refresh_registry(agent), _tolerant_loads)
    return (request, inv.end) if request else None
