"""Stop the model stream after the first complete, translatable </invoke>.

The core stops streaming when the callback returns text (models.py unified_turn), which is how it
already cuts JSON replies. This wraps the response callback; the original always runs first.
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
        state = {"done": False}

        async def wrapped(chunk: str, full: str) -> str | None:
            result = await original(chunk, full)
            if result is not None or state["done"]:
                return result
            if "invoke" not in full[max(0, len(full) - len(chunk) - 12):]:
                return None
            try:
                inv = bridge.find_first_invocation(full)
                if not isinstance(inv, bridge.Invocation):
                    return None  # nothing yet, still being written, or malformed
                state["done"] = True
                config = integration.load_config(agent)
                if not config.get("early_stop", True) or not integration.applies(agent, config):
                    return None
                if integration.translate_response(agent, full[: inv.end]) is None:
                    return None
                agent.loop_data.params_temporary[integration.EARLY_STOP_KEY] = True
                return full[: inv.end].strip()
            except Exception as exc:  # fail open: keep streaming
                state["done"] = True
                PrintStyle(font_color="orange", padding=False).print(
                    f"claude_tool_bridge: stream cut disabled ({type(exc).__name__}: {exc})"
                )
                return None

        call_kwargs["response_callback"] = wrapped
