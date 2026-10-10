import json
import re

import pytest

from usr.plugins.claude_tool_bridge.helpers import bridge

CODE_PARAMS = {
    "runtime": {"type": "string", "enum": ["terminal", "python", "nodejs", "output", "reset"]},
    "code": {"type": "string"},
    "session": {"type": "integer"},
    "reset": {"type": "boolean"},
}
PERMISSIVE = {"type": "object", "properties": {}, "additionalProperties": True}
EDITOR_DESCRIPTION = (
    "text editor\n"
    'Arguments example:\n```json\n{"action": "read", "path": "/p.py", "line_from": 1, "line_to": 50}\n```\n'
    'Arguments example:\n```json\n{"action": "write", "path": "/p.py", "content": "x"}\n```\n'
)
MCP_NAME = "my server.tool x"
MCP_NATIVE = "my_server_tool_x_abcd1234"


def _tool(name, parameters, description=""):
    return {"type": "function", "name": name, "description": description, "parameters": parameters}


@pytest.fixture
def registry():
    tools = [
        _tool("code_execution_tool", {"type": "object", "properties": CODE_PARAMS}),
        _tool("text_editor", PERMISSIVE, EDITOR_DESCRIPTION),
        _tool("response", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        _tool("myserver.context", {"type": "object", "properties": {
            "name": {"type": "string"}, "include_content": {"type": "boolean"}, "limit": {"type": "integer"}}}),
        _tool(MCP_NATIVE, {"type": "object", "properties": {"q": {"type": "string"}}}),
        _tool("multi", {"type": "object", "properties": {"v": {"type": ["integer", "null"]}}}),
    ]
    return bridge.build_registry(tools, {MCP_NATIVE: MCP_NAME})


def invoke(name, params, ns=""):
    body = "".join(f'<{ns}parameter name="{k}">{v}</{ns}parameter>\n' for k, v in params)
    return f'<{ns}invoke name="{name}">\n{body}</{ns}invoke>'


CODE = [("runtime", "terminal"), ("session", "2"), ("reset", "true"), ("code", "ls -la\necho hi")]


def test_single_invocation_types_and_trailing_newline(registry):
    request = bridge.translate("\n" + invoke("code_execution_tool", CODE) + "\n", registry)
    assert request == {
        "tool_name": "code_execution_tool",
        "tool_args": {"runtime": "terminal", "session": 2, "reset": True, "code": "ls -la\necho hi"},
    }


def test_function_calls_wrapper_is_ignored(registry):
    text = "<function_calls>\n" + invoke("code_execution_tool", CODE) + "\n</function_calls>"
    request = bridge.translate(text, registry)
    assert request["tool_args"]["session"] == 2 and "thoughts" not in request


def test_antml_prefix(registry):
    text = "<function_calls>\n" + invoke("code_execution_tool", CODE, "antml:") + "\n</function_calls>"
    assert bridge.translate(text, registry)["tool_name"] == "code_execution_tool"


def test_prose_before_becomes_thoughts(registry):
    text = "First I look.\n\nThen I act.\n\n\n" + invoke("code_execution_tool", CODE)
    assert bridge.translate(text, registry)["thoughts"] == ["First I look.", "Then I act."]


def test_single_tool_args_parameter(registry):
    args = json.dumps({"runtime": "python", "code": "print(1)"})
    request = bridge.translate(invoke("code_execution_tool", [("tool_args", args)]), registry)
    assert request["tool_args"] == {"runtime": "python", "code": "print(1)"}


def test_thoughts_headline_tool_name_tool_args_as_parameters(registry):
    params = [
        ("thoughts", '["a", "b"]'), ("headline", "Doing it"), ("tool_name", "code_execution_tool"),
        ("tool_args", '{"runtime": "terminal", "code": "pwd"}'),
    ]
    request = bridge.translate(invoke("code_execution_tool", params), registry)
    assert request == {
        "thoughts": ["a", "b"], "headline": "Doing it", "tool_name": "code_execution_tool",
        "tool_args": {"runtime": "terminal", "code": "pwd"},
    }


def test_conflicting_tool_name_parameter_is_not_translated(registry):
    params = [("tool_name", "response"), ("code", "x")]
    assert bridge.translate(invoke("code_execution_tool", params), registry) is None


def test_method_suffix_and_mcp_names(registry):
    patch = bridge.translate(invoke("text_editor:patch", [("path", "/a"), ("old_text", "1"), ("new_text", "2")]), registry)
    assert patch["tool_name"] == "text_editor:patch"
    mcp = bridge.translate(invoke("myserver.context", [("name", "f"), ("include_content", "true"), ("limit", "5")]), registry)
    assert mcp["tool_args"] == {"name": "f", "include_content": True, "limit": 5}
    native = bridge.translate(invoke(MCP_NATIVE, [("q", "x")]), registry)
    assert native["tool_name"] == MCP_NAME


def test_unknown_tool_is_not_translated(registry):
    assert bridge.translate(invoke("nope", [("a", "1")]), registry) is None
    assert bridge.translate(invoke("nope:patch", [("a", "1")]), registry) is None


def test_no_params_or_self_closing_is_not_translated(registry):
    assert bridge.translate('<invoke name="code_execution_tool">\n</invoke>', registry) is None
    assert bridge.translate('<invoke name="code_execution_tool"/>', registry) is None


def test_missing_required_is_not_translated(registry):
    assert bridge.translate(invoke("response", [("other", "x")]), registry) is None
    assert bridge.translate(invoke("response", [("text", "done")]), registry)["tool_args"] == {"text": "done"}


def test_loop_takes_only_the_first_valid_invocation(registry):
    text = invoke("code_execution_tool", CODE) + "\n\n" + (invoke("code_execution_tool", CODE) + "\n") * 100
    assert bridge.translate(text, registry)["tool_args"]["code"] == "ls -la\necho hi"


def test_loop_of_empty_invocations_is_not_translated(registry):
    assert bridge.translate('<invoke name="code_execution_tool">\n</invoke>\n' * 100, registry) is None


def test_string_params_that_look_like_json_stay_strings(registry):
    for raw in ("123", "true", "null", '{"a": 1}', "[1, 2]"):
        request = bridge.translate(invoke("code_execution_tool", [("code", raw)]), registry)
        assert request["tool_args"]["code"] == raw
    request = bridge.translate(invoke("text_editor", [("action", "write"), ("path", "/a.json"), ("content", '{"a": 1}')]), registry)
    assert request["tool_args"]["content"] == '{"a": 1}'


def test_types_inferred_from_usage_examples_when_schema_is_permissive(registry):
    params = [("action", "read"), ("path", "/12"), ("line_from", "3"), ("line_to", "9")]
    assert bridge.translate(invoke("text_editor", params), registry)["tool_args"] == {
        "action": "read", "path": "/12", "line_from": 3, "line_to": 9}
    # no schema type and no example: kept as text
    assert bridge.translate(invoke("text_editor", [("action", "read"), ("path", "/a"), ("mystery", "5")]), registry)["tool_args"]["mystery"] == "5"


def test_union_types_and_bad_values(registry):
    assert bridge.translate(invoke("multi", [("v", "7")]), registry)["tool_args"] == {"v": 7}
    assert bridge.translate(invoke("multi", [("v", "seven")]), registry)["tool_args"] == {"v": "seven"}
    assert bridge.translate(invoke("code_execution_tool", [("code", "x"), ("session", "abc")]), registry)["tool_args"]["session"] == "abc"
    assert bridge.translate(invoke("code_execution_tool", [("code", "x"), ("reset", "False")]), registry)["tool_args"]["reset"] is False


def test_literal_closing_parameter_tag_inside_code(registry):
    code = 'print("</parameter> is a tag")\nprint(2)'
    request = bridge.translate(invoke("code_execution_tool", [("runtime", "python"), ("code", code)]), registry)
    assert request["tool_args"]["code"] == code


def test_residue_in_body_is_not_translated(registry):
    text = '<invoke name="code_execution_tool">\nstray words\n<parameter name="code">x</parameter>\n</invoke>'
    assert bridge.translate(text, registry) is None


def test_invoke_inside_code_fence_or_mid_line_is_quoted_not_called(registry):
    call = invoke("code_execution_tool", CODE)
    assert bridge.translate("```xml\n" + call + "\n```", registry) is None
    assert bridge.translate("~~~\n" + call + "\n~~~", registry) is None
    assert bridge.translate("Use it like this: " + call, registry) is None
    assert bridge.translate("```\nexample\n```\n" + call, registry) is not None


def test_garbage_after_the_invocation_is_ignored(registry):
    text = invoke("code_execution_tool", CODE) + "\n```\n\nSorry, fixing the format:\n</tool_call>\n</invoke>\n</function_calls>"
    assert bridge.translate(text, registry)["tool_name"] == "code_execution_tool"


def test_repeated_parameter_last_wins(registry):
    request = bridge.translate(invoke("code_execution_tool", [("code", "a"), ("code", "b")]), registry)
    assert request["tool_args"]["code"] == "b"


def test_incomplete_invocation_is_reported_while_streaming(registry):
    full = invoke("code_execution_tool", CODE)
    for cut in (40, 60, len(full) - 5, len(full) - 1):
        assert bridge.find_first_invocation(full[:cut]) is bridge.INCOMPLETE, cut
    assert isinstance(bridge.find_first_invocation(full), bridge.Invocation)
    assert bridge.find_first_invocation("no call here") is None


def test_invocation_end_marks_the_cut_point(registry):
    full = "Intro.\n" + invoke("code_execution_tool", CODE) + "\nmore text after"
    inv = bridge.find_first_invocation(full)
    assert full[: inv.end].endswith("</invoke>") and inv.prose == "Intro."


def test_non_utf_text_and_empty_inputs(registry):
    assert bridge.translate("", registry) is None
    assert bridge.translate(None, registry) is None


@pytest.mark.parametrize(
    "provider,name,expected",
    [
        ("anthropic", "claude-sonnet-5-5", True),
        ("Anthropic", "Claude-Opus-5-5", True),
        ("openrouter", "anthropic/claude-sonnet-5-5", False),
        ("anthropic", "gpt-5", False),
        ("openai", "claude-x", False),
    ],
)
def test_model_matches(provider, name, expected):
    assert bridge.model_matches(provider, name, ["anthropic"], ["claude-*"]) is expected


def test_model_matches_last_segment_and_open_filters():
    assert bridge.model_matches("openrouter", "anthropic/claude-sonnet-5-5", ["openrouter"], ["claude-*"])
    assert bridge.model_matches("x", "y", [], [])


def test_registry_ignores_junk_tools():
    registry = bridge.build_registry([{"name": ""}, {"name": "t", "parameters": None, "description": None}])
    assert list(registry.tools) == ["t"] and registry.resolve("t") is not None
    assert re.fullmatch(r"\w+", "t")


# ---------------------------------------------------------------- namespace prefix, StreamWatch, stop

NS = "antml" + ":"


def test_half_namespaced_closing_tag_is_still_incomplete(registry):
    head = '<invoke name="code_execution_tool">\n<parameter name="code">x</parameter>\n'
    for tail in ("</" + NS + "inv", "<" + NS + "par", "</a", "<antml"):
        assert bridge.find_first_invocation(head + tail) is bridge.INCOMPLETE, tail
    assert bridge.find_first_invocation('<invoke name="t">\nstray words\n') is None  # not a list of parameters


def watch_all(text, size=8):
    watch = bridge.StreamWatch()
    for i in range(0, len(text), size):
        inv = watch.feed(text[i : i + size], text[: i + size])
        if inv is not None:
            return inv
    return None


@pytest.fixture
def parses(monkeypatch):
    calls = {"n": 0}
    original = bridge._scan

    def counting(text):
        calls["n"] += 1
        return original(text)

    monkeypatch.setattr(bridge, "_scan", counting)
    return calls


@pytest.mark.parametrize("ns", ["", NS])
@pytest.mark.parametrize("size", [1, 3, 7, 8, 64, 500])
def test_stream_watch_finds_the_same_invocation_as_parsing_every_chunk(registry, ns, size):
    text = "<function_calls>\n" + invoke("code_execution_tool", CODE, ns) + "\n" + invoke("code_execution_tool", CODE, ns)
    expected = bridge.find_first_invocation(text)
    found = watch_all(text, size)
    assert isinstance(found, bridge.Invocation) and found.end == expected.end and found.params == expected.params
    assert text[: found.end].endswith("</" + ns + "invoke>")


def test_stream_watch_reports_the_first_invocation_once_even_when_it_is_empty(registry):
    watch = bridge.StreamWatch()
    text = '<invoke name="code_execution_tool">\n</invoke>\n' + invoke("code_execution_tool", CODE)
    results = [watch.feed(text[i : i + 5], text[: i + 5]) for i in range(0, len(text), 5)]
    found = [r for r in results if r is not None]
    assert len(found) == 1 and found[0].params == []  # the later full call is not offered: the model may correct itself


def test_stream_watch_does_not_parse_a_long_value_that_only_mentions_invoke(registry, parses):
    code = "def f(x):\n    return invoke(x)  # </parameter> literal\n" * 2000
    text = invoke("text_editor", [("action", "write"), ("path", "/a.py"), ("content", code)])
    found = watch_all(text)
    assert found is not None and found.end == len(text)
    assert parses["n"] <= 4, parses


def test_stream_watch_gives_up_on_a_broken_call_without_parsing_the_runaway_that_follows(parses):
    text = '<invoke name="x">\nstray words\n' + 'noise <invoke name="y">\n</invoke>\n' * 1500
    assert watch_all(text) is None
    assert parses["n"] <= 12, parses


def test_stream_watch_ignores_quoted_examples_and_fences(parses):
    prose = ("Use `" + '<invoke name="x">` inline, the word invoke appears a lot.\n') * 1500
    fenced = "```xml\n" + '<invoke name="x">\n<parameter name="a">1</parameter>\n</invoke>\n' + "```\n"
    assert watch_all(prose) is None and parses["n"] == 0
    assert watch_all(fenced) is None and parses["n"] <= 3, parses


def test_stream_watch_markup_in_a_value_does_not_cause_a_parse_per_tag(registry, parses):
    html = "<br/><img src='a'/>\n" * 3000
    text = invoke("text_editor", [("action", "write"), ("path", "/a.html"), ("content", html)])
    assert watch_all(text) is not None and parses["n"] <= 4, parses


def test_stream_watch_stops_looking_when_its_budget_runs_out():
    watch = bridge.StreamWatch(budget=60)
    head = '<invoke name="code_execution_tool">\n<parameter name="code">'
    assert watch.feed(head, head) is None and not watch.done and watch.pending
    full = head + "x" * 50 + "</parameter>\n</invoke>"
    assert watch.feed(full[len(head):], full) is None and watch.done
    assert watch.feed("more", full + "more") is None


# ---- the stop sequence the provider is asked to honour


@pytest.mark.parametrize("ns", ["", NS])
def test_a_stopped_reply_is_completed_into_the_same_call(registry, ns):
    full = "Thinking.\n\n" + invoke("code_execution_tool", CODE, ns) + "\nthe model went on writing"
    cut = min(full.index(s) for s in bridge.STOP_SEQUENCES if s in full)
    stopped = full[:cut]  # what the provider returns: everything before the stop sequence
    assert stopped.endswith("</" + ns + "paramete")
    assert bridge.find_first_invocation(stopped) is bridge.INCOMPLETE
    completed = bridge.complete_stopped(stopped)
    assert bridge.translate(completed, registry) == bridge.translate(full, registry)
    assert bridge.translate(completed, registry)["tool_args"]["code"] == "ls -la\necho hi"


def test_complete_stopped_leaves_every_other_reply_alone():
    for text in ("", "plain", "x</parameter>", "x</parameter>\n</invoke>", "x</paramete\n", "x</paramet", '{"tool_name": "x"}'):
        assert bridge.complete_stopped(text) is None, text


def test_stop_sequences_cannot_match_an_empty_or_unrelated_call():
    for text in (
        '<invoke name="reader">\n</invoke>', '<invoke name="a"/>', "plain </invoke>", "a>\n</invoke>",
        '{"tool_name": "response", "tool_args": {"text": "a\n</invoke>"}}',
    ):
        assert not any(s in text for s in bridge.STOP_SEQUENCES), text


def test_a_literal_closing_pair_inside_a_value_ends_the_call_there(registry):
    # documented limit, the same with or without a stop sequence: the first "</parameter>\n</invoke>" is the end
    code = "print('</parameter>\n</invoke>')\nprint(2)"
    request = bridge.translate(invoke("code_execution_tool", [("code", code)]), registry)
    assert request["tool_args"]["code"] == "print('"
