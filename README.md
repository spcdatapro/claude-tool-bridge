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
- **Stops the model** after the first call, so it does not keep writing calls it has not seen results for.
  Two mechanisms, see [Stopping early](#stopping-early).
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

Settings → Agent → Claude Tool Bridge, or `default_config.yaml` (a saved `config.json` overrides it):

| Key | Default | Meaning |
| --- | --- | --- |
| `providers` | `[anthropic]` | Chat-model providers the bridge applies to |
| `model_patterns` | `["claude-*"]` | `fnmatch` patterns on the chat model name (also tried on its last `/` segment) |
| `early_stop` | `true` | Ask the core to stop streaming after the first translatable `</invoke>` |
| `stop_sequence` | `true` | Ask the provider to stop generating at the end of the first call (see below) |
| `log_translations` | `true` | One console line per translation (tool name and argument count, never content) |

## Stopping early

Claude tends to keep writing after a call: more calls, invented results, sometimes until `max_tokens`. Everything
after the first call is discarded, but you wait for it and pay for it.

- `early_stop` makes the plugin's stream callback ask Agent Zero to stop reading as soon as the first call is
  complete. **In Agent Zero 2.13 this does not shorten the wait**: the core's `_context_window` plugin wraps the
  callback to read the provider's token usage and its wrapper never passes the stop request on, so the stream is
  read until the model finishes. Without that plugin it works.
- `stop_sequence` asks the provider itself to stop (Anthropic `stop_sequences`, set on the request). The
  provider ends the reply at the end of the first call, so the wait and the output tokens after it disappear and the
  usage numbers stay exact. The reply comes back ending in a half closing tag, which the plugin recognises and
  closes again before translating. It is on by default; turn it off in Settings → Agent → Claude Tool Bridge.

Safeguards for `stop_sequence`: the reply is only completed when it ends in that half tag, and never when it ran out
of `max_tokens`. A stop also cuts off anything the model would have written after a call, including a correction of a
broken one. When a stopped reply holds no usable call, the plugin drops it and Agent Zero asks again
without a stop sequence (no warning, nothing added to the history), and the stop sequence stays paused for the next three turns.
The price is one extra, short request in those cases (about 1 in 10 replies written as XML in our tests).

## Installation

Copy or clone this folder to `usr/plugins/claude_tool_bridge` (the folder name must use underscores) and
enable it in the plugin list. Restart Agent Zero if a module of the plugin was added after it started.

## Limits

- Degenerate loops (thousands of empty `<invoke>` tags) are not rescued; they are bounded by `max_tokens`.
- Plain prose is not turned into a `response` call.
- When a reply holds a complete `<invoke>` followed by a corrected JSON call, the XML call runs (the reply is
  cut at its `</invoke>`, or stopped there with `stop_sequence`). Set `early_stop: false` and set `stop_sequence: false` to let
  the JSON win.
- A parameter value that itself contains the text `</parameter>` followed by a line break and `</invoke>` ends the call there,
  with or without a stop sequence.
- This is a translation layer. The structural fix is native tool calling in Agent Zero's Anthropic transport.

## Development

```bash
pytest usr/plugins/claude_tool_bridge/tests -q    # inside the Agent Zero framework environment
```

`helpers/bridge.py` is pure standard-library code; `helpers/integration.py` and the three extensions are
the only parts that import Agent Zero.

Cost: a fraction of a millisecond of CPU per streamed reply. The plugin settings and the tool list are cached for a few seconds and two
minutes, and a reply is parsed only when a tag that can complete the first call arrives, with a hard cap on work per reply.

## License

MIT
