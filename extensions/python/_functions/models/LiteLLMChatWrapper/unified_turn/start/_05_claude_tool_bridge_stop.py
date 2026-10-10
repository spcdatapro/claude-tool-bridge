"""Ask the provider to stop generating right after the first tool call (opt-in: stop_sequence).

Cutting the stream from the client does not work in Agent Zero 2.13: _context_window wraps the response
callback (_10_drain_provider_usage) to read the provider's token usage, and its wrapper never asks the
core to stop. A stop sequence is applied by the provider, so the reply ends there and the usage is complete.

The reply then ends in a half closing tag that bridge.complete_stopped() recognises and closes again.
Runs before _context_window's extension so it still sees the plugin's callback.
"""

from __future__ import annotations

from typing import Any

from helpers.extension import Extension
from usr.plugins.claude_tool_bridge.helpers import bridge


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


class ClaudeToolBridgeStop(Extension):
    def execute(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not isinstance(data, dict):
            return
        call_kwargs = data.get("kwargs")
        if not isinstance(call_kwargs, dict):
            return
        state = getattr(call_kwargs.get("response_callback"), "ctb_state", None)
        if not isinstance(state, dict) or not state.get("stop"):
            return
        args = data.get("args")
        model = args[0] if isinstance(args, tuple) and args else None
        conf = getattr(model, "a0_model_conf", None)
        if conf is None or not bridge.model_matches(
            str(getattr(conf, "provider", "")),
            str(getattr(conf, "name", "")),
            state.get("providers"),
            state.get("model_patterns"),
        ):
            state["stop"] = False
            return
        model_kwargs = getattr(model, "kwargs", None)
        model_kwargs = model_kwargs if isinstance(model_kwargs, dict) else {}
        stops = _as_list(call_kwargs.get("stop", model_kwargs.get("stop")))
        stops += [s for s in bridge.STOP_SEQUENCES if s not in stops]
        call_kwargs["stop"] = stops
        for key in ("max_tokens", "max_completion_tokens"):
            limit = call_kwargs.get(key, model_kwargs.get(key))
            if isinstance(limit, int) and not isinstance(limit, bool):
                state["max_tokens"] = limit
                break
