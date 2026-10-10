"""Watch the model stream for the end of the first <invoke> and ask the core to stop there.

The core stops streaming when the callback returns text (models.py unified_turn), which is how it
already cuts JSON replies. This wraps the response callback; the original always runs first. The
wrapper also carries this turn's state to the stop extension that builds the request.
"""

from __future__ import annotations

from typing import Any

from helpers.extension import Extension
from helpers.print_style import PrintStyle
from usr.plugins.claude_tool_bridge.helpers import bridge, integration


class ClaudeToolBridgeStream(Extension):
    def execute(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.agent or not isinstance(data, dict):
            return
        call_kwargs = data.get("kwargs")
        if not isinstance(call_kwargs, dict):
            return
        original = call_kwargs.get("response_callback")
        if original is None:
            return
        agent = self.agent
        try:
            config, applies = integration.turn_plan(agent)
            if not applies:
                return
            state = integration.turn_state(agent)
            state["stop"] = integration.stop_enabled(agent, config)
            state["providers"] = config.get("providers")
            state["model_patterns"] = config.get("model_patterns")
        except Exception as exc:  # fail open: leave the callback alone
            PrintStyle(font_color="orange", padding=False).print(
                f"claude_tool_bridge: stream watch disabled ({type(exc).__name__}: {exc})"
            )
            return
        early_stop = bool(config.get("early_stop", True))
        watch = bridge.StreamWatch()

        async def wrapped(chunk: str, full: str) -> str | None:
            result = await original(chunk, full)
            if result is not None or not early_stop:
                return result
            try:
                inv = watch.feed(chunk, full)
                if inv is None:
                    return None  # nothing yet, still being written, or malformed
                if integration.translate_response(agent, full[: inv.end]) is None:
                    return None
                agent.loop_data.params_temporary[integration.EARLY_STOP_KEY] = True
                return full[: inv.end].strip()
            except Exception as exc:  # fail open: keep streaming
                watch.done = True
                PrintStyle(font_color="orange", padding=False).print(
                    f"claude_tool_bridge: stream cut disabled ({type(exc).__name__}: {exc})"
                )
                return None

        wrapped.ctb_state = state  # type: ignore[attr-defined]
        call_kwargs["response_callback"] = wrapped
