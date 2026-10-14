"""Rewrite a Claude <invoke> reply as Agent Zero JSON, before _context_doctor sees it.

Also throws away replies that are a loop (so the turn is asked again) and sends a reply that is only prose
through the `response` tool, which is what the model meant and what it would send next anyway.
"""

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
        has_invoke = "invoke" in response
        stopped_tail = bridge.complete_stopped(response) is not None
        if not has_invoke and not stopped_tail and response.lstrip()[:1] in ("{", ""):
            # a JSON reply, or none at all: nothing to do (only reset the retries if a usable reply ends them)
            if integration.retry_pending(self.agent) and integration.has_json_tool_request(response):
                integration.retry_reset(self.agent)
            return

        try:
            config, applies = integration.turn_plan(self.agent)
            if not applies:
                return
            state = integration.turn_state(self.agent)
            limit_hit = integration.hit_output_limit(llm_result, state)
            asked = bool(state.get("stop")) and not limit_hit
            guarded = bool(state.get("guard"))
            text, completed = response, None
            if asked:
                completed = bridge.complete_stopped(response)  # the provider stopped it at the end of the call
                if completed is not None:
                    text = completed
            outcome = integration.translate_response(self.agent, text) if (has_invoke or completed) else None
            if outcome is not None:
                request, _ = outcome
                llm_result.response = json.dumps(request, ensure_ascii=False)
                integration.retry_reset(self.agent)
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
                return
            if integration.retry_pending(self.agent) and integration.has_json_tool_request(response):
                integration.retry_reset(self.agent)  # the model recovered on its own
                return

            if completed is not None:
                # the stop cost the rest of a reply that had no usable call (the model might have
                # corrected itself after it): drop the reply so the core asks again, this time without a stop
                self._ask_again(result_data, llm_result, "stopped reply held no usable call", pause=True)
                return
            if guarded and ((not limit_hit and bridge.loop_cut(response)) or bridge.looks_degenerate(response)):
                if not integration.has_json_tool_request(response):  # a call can be repetitive too: not a loop
                    self._ask_again(result_data, llm_result, "reply was a loop", pause=False)
                return
            if (asked or guarded) and not limit_hit and bridge.find_first_invocation(response) is bridge.INCOMPLETE:
                # a stop was requested and the reply ends inside a call, but not where the stop sequence
                # leaves it: the provider cut it some other way (a stop string inside a value, most likely)
                if not integration.has_json_tool_request(response):
                    self._ask_again(result_data, llm_result, "reply cut inside a call at an unexpected place", pause=True)
                return

            if config.get("prose_as_response", True) and not limit_hit:
                answer = integration.plain_text_answer(self.agent, response)
                if answer is not None:
                    llm_result.response = json.dumps(answer, ensure_ascii=False)
                    integration.retry_reset(self.agent)
                    if config.get("log_translations", True):
                        PrintStyle(font_color="cyan", padding=False).print(
                            "claude_tool_bridge: plain text -> response "
                            f"({len(answer['tool_args']['text'])} chars)"
                        )
        except Exception as exc:  # never break the turn: fall back to stock behaviour
            PrintStyle(font_color="orange", padding=False).print(
                f"claude_tool_bridge: translation skipped ({type(exc).__name__}: {exc})"
            )

    def _ask_again(self, result_data: dict[str, Any], llm_result: Any, why: str, pause: bool) -> None:
        """Make the core run this turn again: an empty reply makes _context_doctor step aside, and
        skip_default_processing makes the loop continue without touching the history. pause also turns the
        stop sequences off for a few turns (the next request must be able to say what the stop cut)."""
        if pause:
            integration.stop_failed(self.agent)
        if not integration.allow_silent_retry(self.agent):
            PrintStyle(font_color="orange", padding=False).print(
                f"claude_tool_bridge: {why}; not asking again (already did {integration.SILENT_RETRY_LIMIT} times)"
            )
            return
        llm_result.response = ""
        result_data["skip_default_processing"] = True
        PrintStyle(font_color="orange", padding=False).print(
            f"claude_tool_bridge: {why}; asking again"
            + (f" without stop sequences (paused for {integration.STOP_PAUSE_TURNS} turns)" if pause else "")
        )
