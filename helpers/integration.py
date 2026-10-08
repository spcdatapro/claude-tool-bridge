"""Glue between the pure bridge and Agent Zero, shared by the plugin's extensions."""

from __future__ import annotations

from typing import Any

from helpers import extract_tools, plugins
from usr.plugins.claude_tool_bridge.helpers import bridge

PLUGIN_NAME = "claude_tool_bridge"
REGISTRY_KEY = "claude_tool_bridge.registry"
EARLY_STOP_KEY = "claude_tool_bridge.early_stop"

DEFAULTS: dict[str, Any] = {
    "providers": ["anthropic"],
    "model_patterns": ["claude-*"],
    "early_stop": True,
    "log_translations": True,
}


def load_config(agent: Any) -> dict[str, Any]:
    """Plugin config over the defaults (a saved config.json replaces default_config.yaml whole)."""
    config = dict(DEFAULTS)
    saved = plugins.get_plugin_config(PLUGIN_NAME, agent=agent)
    if isinstance(saved, dict):
        config.update(saved)
    return config


def applies(agent: Any, config: dict[str, Any]) -> bool:
    """True if the agent's chat model is covered by the configured providers and patterns."""
    try:
        from plugins._model_config.helpers import model_config

        chat = model_config.get_chat_model_config(agent) or {}
        return bridge.model_matches(
            str(chat.get("provider") or ""),
            str(chat.get("name") or ""),
            config.get("providers"),
            config.get("model_patterns"),
        )
    except Exception:
        return False


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


def get_registry(agent: Any) -> bridge.Registry:
    """Tool registry for this turn; built once, only when an <invoke> shows up."""
    params = agent.loop_data.params_temporary
    registry = params.get(REGISTRY_KEY)
    if registry is None:
        from helpers import responses_tools

        tools, name_map = responses_tools.build_responses_function_tools(agent)
        registry = bridge.build_registry(tools, name_map)
        params[REGISTRY_KEY] = registry
    return registry


def translate_response(agent: Any, text: str) -> tuple[dict[str, Any], int] | None:
    """(request, end of the first </invoke>) or None. Never translates a reply that already
    carries a JSON call outside the invocation: _context_doctor rescues those."""
    inv = bridge.find_first_invocation(text)
    if not isinstance(inv, bridge.Invocation):
        return None
    if has_json_tool_request(text[: inv.start] + "\n" + text[inv.end:]):
        return None
    request = bridge.request_from_invocation(inv, get_registry(agent))
    return (request, inv.end) if request else None
