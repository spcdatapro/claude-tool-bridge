"""Rewrite a Claude <invoke> reply as Agent Zero JSON, before _context_doctor sees it."""

from __future__ import annotations

import json
from typing import Any

from helpers.extension import Extension
from helpers.print_style import PrintStyle
from usr.plugins.claude_tool_bridge.helpers import bridge, integration


class ClaudeToolBridge(Extension):
    def execute(self, result_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.agent or not isinstance(result_data, dict):
            return
        if result_data.get("skip_default_processing"):
            return

        llm_result = result_data.get("llm_result")
        if getattr(llm_result, "function_calls", None) or getattr(llm_result, "mode", None) == "responses":
            return
        response = getattr(llm_result, "response", None)
        if not isinstance(response, str):
            return
        if "invoke" not in response and bridge.complete_stopped(response) is None:
            return  # no call in sight (a stopped reply can end before any "invoke" was written)

        try:
            config, applies = integration.turn_plan(self.agent)
            if not applies:
                return
            state = integration.turn_state(self.agent)
            text, completed = response, None
            asked = bool(state.get("stop")) and not integration.hit_output_limit(llm_result, state)
            if asked:
                completed = bridge.complete_stopped(response)  # the provider stopped it at the end of the call
                if completed is not None:
                    text = completed
            outcome = integration.translate_response(self.agent, text)
            if outcome is None:
                if completed is not None:
                    # the stop cost the rest of a reply that had no usable call (the model might have
                    # corrected itself after it): drop the reply so the core asks again, this time without a stop
                    self._retry_without_stop(result_data, llm_result, "stopped reply held no usable call")
                elif asked and bridge.find_first_invocation(response) is bridge.INCOMPLETE:
                    # a stop was requested and the reply ends inside a call, but not where the stop
                    # sequence leaves it: the provider cut it some other way
                    self._retry_without_stop(result_data, llm_result, "reply cut inside a call at an unexpected place")
                return
            request, _ = outcome
            llm_result.response = json.dumps(request, ensure_ascii=False)
            if config.get("log_translations", True):
                params = getattr(getattr(self.agent, "loop_data", None), "params_temporary", {}) or {}
                how = (
                    " (stop sequence)" if completed is not None
                    else " (early stop requested)" if params.get(integration.EARLY_STOP_KEY) else ""
                )
                PrintStyle(font_color="cyan", padding=False).print(
                    f"claude_tool_bridge: <invoke> -> {request['tool_name']} "
                    f"with {len(request['tool_args'])} args{how}"
                )
        except Exception as exc:  # never break the turn: fall back to stock behaviour
            PrintStyle(font_color="orange", padding=False).print(
                f"claude_tool_bridge: translation skipped ({type(exc).__name__}: {exc})"
            )

    def _retry_without_stop(self, result_data: dict[str, Any], llm_result: Any, why: str) -> None:
        """Make the core run this turn again: an empty reply makes _context_doctor step aside, and
        skip_default_processing makes the loop continue without touching the history."""
        integration.stop_failed(self.agent)
        llm_result.response = ""
        result_data["skip_default_processing"] = True
        PrintStyle(font_color="orange", padding=False).print(
            f"claude_tool_bridge: {why}; asking again without a stop sequence "
            f"(paused for {integration.STOP_PAUSE_TURNS} turns)"
        )
