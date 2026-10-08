"""Rewrite a Claude <invoke> reply as Agent Zero JSON, before _context_doctor sees it."""

from __future__ import annotations

import json
from typing import Any

from helpers.extension import Extension
from helpers.print_style import PrintStyle
from usr.plugins.claude_tool_bridge.helpers import integration


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
        if not isinstance(response, str) or "invoke" not in response:
            return

        try:
            config = integration.load_config(self.agent)
            if not integration.applies(self.agent, config):
                return
            outcome = integration.translate_response(self.agent, response)
            if outcome is None:
                return
            request, _ = outcome
            llm_result.response = json.dumps(request, ensure_ascii=False)
            if config.get("log_translations", True):
                params = getattr(getattr(self.agent, "loop_data", None), "params_temporary", {}) or {}
                stopped = " (stream cut)" if params.get(integration.EARLY_STOP_KEY) else ""
                PrintStyle(font_color="cyan", padding=False).print(
                    f"claude_tool_bridge: <invoke> -> {request['tool_name']} "
                    f"with {len(request['tool_args'])} args{stopped}"
                )
        except Exception as exc:  # never break the turn: fall back to stock behaviour
            PrintStyle(font_color="orange", padding=False).print(
                f"claude_tool_bridge: translation skipped ({type(exc).__name__}: {exc})"
            )
