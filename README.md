# Claude Tool Bridge

An [Agent Zero](https://github.com/agent0ai/agent-zero) plugin that translates Claude's native tool-call
syntax into the JSON format Agent Zero expects, so Claude 5 models (for example `claude-sonnet-5-5`) work
reliably in Agent Zero's chat mode.

## The problem

Agent Zero talks to Anthropic in chat mode, without native tools, so the model has to write each call as
`{"tool_name": ..., "tool_args": {...}}`. Some Claude models keep writing their own native syntax instead:

```xml
<invoke name="code_execution_tool">
<parameter name="runtime">terminal</parameter>
<parameter name="code">ls -la</parameter>
</invoke>
```

Agent Zero treats that reply as unusable, retries, and after N consecutive unusable replies (Settings →
Agent) it stops the agent. In a replay of 20 real prompts, `claude-sonnet-5-5` produced an unusable reply
in 45–90 % of the turns depending on its thinking settings. Almost all of them were correct calls in the wrong syntax.

## What it does

- **Translates** the first complete `<invoke>` of a reply into an Agent Zero JSON tool request before
  `_context_doctor` and the rest of the message loop see it. Text written before the call becomes `thoughts`.
- **Types the arguments** from the tool's schema, falling back to the types in Agent Zero's own usage
  examples. A string parameter that happens to look like JSON (`"123"`, `{"a": 1}`) stays a string.
- **Stops the stream** at the first translatable `</invoke>`, so the model does not keep writing calls it
  has not seen results for (disable with `early_stop: false`).
- **Understands** `tool:method` names, MCP tools, `antml:` prefixes, a `<function_calls>` wrapper, and
  `thoughts` / `headline` / `tool_args` written as parameters.

It deliberately does **nothing** when:

- the reply already contains a JSON tool call (Agent Zero or `_context_doctor` handles it);
- the `<invoke>` is quoted: inside a code fence, or not at the start of a line;
- the tool is unknown, a required argument is missing, or the invocation has no arguments;
- the chat model is not covered by the configuration, or the call is a native/Responses tool call.

Only the first invocation is used: later ones were written without seeing any result. If anything in the
plugin fails, the reply is left untouched and Agent Zero behaves as it does without the plugin.

## Configuration

`default_config.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `providers` | `[anthropic]` | Chat-model providers the bridge applies to |
| `model_patterns` | `["claude-*"]` | `fnmatch` patterns on the chat model name (also tried on its last `/` segment) |
| `early_stop` | `true` | Stop streaming after the first translatable `</invoke>` |
| `log_translations` | `true` | One console line per translation (tool name and argument count, never content) |

## Installation

Copy or clone this folder to `usr/plugins/claude_tool_bridge` (the folder name must use underscores) and
enable it in the plugin list. Restart Agent Zero if a module of the plugin was added after it started.

## Limits

- Degenerate loops (thousands of empty `<invoke>` tags) are not rescued; they are bounded by `max_tokens`.
- Plain prose is not turned into a `response` call.
- When a reply holds a complete `<invoke>` followed by a corrected JSON call, the stream is cut at the
  `</invoke>` and the XML call runs. Set `early_stop: false` to let the JSON win.
- This is a translation layer. The structural fix is native tool calling in Agent Zero's Anthropic transport.

## Development

```bash
pytest usr/plugins/claude_tool_bridge/tests -q    # inside the Agent Zero framework environment
```

`helpers/bridge.py` is pure standard-library code; `helpers/integration.py` and the two extensions are
the only parts that import Agent Zero.

## License

MIT
