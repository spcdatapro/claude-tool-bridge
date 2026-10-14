import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers import extract_tools
from plugins._context_doctor.helpers.context_doctor import looks_like_tool_call, transform_response
from usr.plugins.claude_tool_bridge.helpers import bridge, integration

ROOT = Path(__file__).resolve().parents[1]
LOOP_EXT = ROOT / "extensions/python/message_loop_result/_05_claude_tool_bridge.py"
STREAM_EXT = ROOT / "extensions/python/_functions/agent/Agent/call_chat_model_turn/start/_05_claude_tool_bridge_stream.py"
STOP_EXT = ROOT / "extensions/python/_functions/models/LiteLLMChatWrapper/unified_turn/start/_05_claude_tool_bridge_stop.py"

TOOLS = [
    {"name": "code_execution_tool", "description": "", "parameters": {"type": "object", "properties": {
        "runtime": {"type": "string"}, "code": {"type": "string"}, "session": {"type": "integer"}}}},
]
CALL = (
    '<invoke name="code_execution_tool">\n<parameter name="runtime">terminal</parameter>\n'
    '<parameter name="session">3</parameter>\n<parameter name="code">ls\npwd</parameter>\n</invoke>'
)


def load(path):
    spec = importlib.util.spec_from_file_location("ext_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent(monkeypatch):
    agent = SimpleNamespace(loop_data=SimpleNamespace(params_temporary={}, params_persistent={}))
    registry = bridge.build_registry(TOOLS)
    monkeypatch.setattr(integration, "get_registry", lambda a: registry)
    monkeypatch.setattr(integration, "load_config", lambda a: dict(integration.DEFAULTS))
    monkeypatch.setattr(integration, "applies", lambda a, c: True)
    return agent


def run_loop_ext(agent, response, **result_fields):
    llm = SimpleNamespace(response=response, function_calls=None, mode="chat", **result_fields)
    result_data = {"llm_result": llm}
    load(LOOP_EXT).ClaudeToolBridge(agent=agent).execute(result_data=result_data)
    return llm, result_data


def test_rewrites_response_and_context_doctor_accepts_it(agent):
    llm, data = run_loop_ext(agent, "Checking.\n\n" + CALL)
    request = json.loads(llm.response)
    assert request["tool_name"] == "code_execution_tool"
    assert request["tool_args"] == {"runtime": "terminal", "session": 3, "code": "ls\npwd"}
    assert request["thoughts"] == ["Checking."]
    assert "skip_default_processing" not in data
    # the real _context_doctor must now see a usable call, and the core must extract it
    transformed = transform_response(llm.response, suppress_xml=True, split_thoughts=True)
    assert looks_like_tool_call(llm.response, transformed)
    name, args = extract_tools.normalize_tool_request(extract_tools.extract_tool_request(transformed))
    assert name == "code_execution_tool" and args["session"] == 3


def test_stock_xml_without_the_bridge_is_what_context_doctor_discards():
    transformed = transform_response(CALL, suppress_xml=True, split_thoughts=True)
    assert not looks_like_tool_call(CALL, transformed)


@pytest.mark.parametrize(
    "response",
    [
        CALL + '\n```\n\nSorry, fixing the format:\n{"tool_name": "response", "tool_args": {"text": "hi"}}',
        '{"tool_name": "code_execution_tool", "tool_args": {"code": "echo \'<invoke name=\\"x\\">\'"}}',
        "plain prose without any call",
        "",
    ],
)
def test_leaves_json_replies_and_prose_alone(agent, response):
    llm, _ = run_loop_ext(agent, response)
    assert llm.response == response


def test_invoke_inside_a_code_string_does_not_hide_a_translatable_call(agent):
    # JSON-looking text inside the <parameter> is part of the call, not a competing JSON call
    code = '{"tool_name": "response", "tool_args": {"text": "x"}}'
    call = f'<invoke name="code_execution_tool">\n<parameter name="code">{code}</parameter>\n</invoke>'
    llm, _ = run_loop_ext(agent, call)
    assert json.loads(llm.response)["tool_args"]["code"] == code


def test_skips_native_calls_responses_mode_handled_results_and_other_models(agent, monkeypatch):
    ext = load(LOOP_EXT).ClaudeToolBridge(agent=agent)
    for llm in (
        SimpleNamespace(response=CALL, function_calls=[object()], mode="chat"),
        SimpleNamespace(response=CALL, function_calls=None, mode="responses"),
    ):
        ext.execute(result_data={"llm_result": llm})
        assert llm.response == CALL
    handled = SimpleNamespace(response=CALL, function_calls=None, mode="chat")
    ext.execute(result_data={"llm_result": handled, "skip_default_processing": True})
    assert handled.response == CALL
    monkeypatch.setattr(integration, "applies", lambda a, c: False)
    llm, _ = run_loop_ext(agent, CALL)
    assert llm.response == CALL


def test_a_failure_never_breaks_the_turn(agent, monkeypatch):
    def boom(a):
        raise RuntimeError("boom")

    monkeypatch.setattr(integration, "get_registry", boom)
    llm, _ = run_loop_ext(agent, CALL)
    assert llm.response == CALL


async def feed(wrapped, text, size=7):
    sent, result = "", None
    for i in range(0, len(text), size):
        chunk = text[i : i + size]
        sent += chunk
        result = await wrapped(chunk, sent)
        if result is not None:
            return result, sent
    return None, sent


def make_wrapped(agent, original):
    data = {"args": (agent,), "kwargs": {"response_callback": original}}
    load(STREAM_EXT).ClaudeToolBridgeStream(agent=agent).execute(data=data)
    return data["kwargs"]["response_callback"]


@pytest.mark.asyncio
async def test_stream_cuts_after_the_first_complete_invocation_split_across_chunks(agent):
    seen = []

    async def original(chunk, full):
        seen.append(chunk)
        return None

    wrapped = make_wrapped(agent, original)
    text = "Thinking.\n" + CALL + "\n" + CALL
    for size in (1, 5, 7, 64):
        agent.loop_data.params_temporary.clear()
        wrapped = make_wrapped(agent, original)
        result, sent = await feed(wrapped, text, size)
        assert result == ("Thinking.\n" + CALL), size
        assert agent.loop_data.params_temporary[integration.EARLY_STOP_KEY] is True
        assert text.startswith(sent) and len(sent) < len(text)
    assert seen


@pytest.mark.asyncio
async def test_stream_original_callback_wins_and_runs_first(agent):
    async def original(chunk, full):
        return "stop-now" if "terminal" in full else None

    result, _ = await feed(make_wrapped(agent, original), CALL, 5)
    assert result == "stop-now"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '<invoke name="code_execution_tool">\n</invoke>\n' + CALL,  # empty first call: no cut, model may correct itself
        '{"tool_name": "response", "tool_args": {"text": "plain json"}}',
        "Talking about an invoke without calling anything.",
    ],
)
async def test_stream_does_not_cut(agent, text):
    async def original(chunk, full):
        return None

    result, sent = await feed(make_wrapped(agent, original), text, 6)
    assert result is None and sent == text


@pytest.mark.asyncio
async def test_stream_disabled_by_config_and_failures_fail_open(agent, monkeypatch):
    async def original(chunk, full):
        return None

    monkeypatch.setattr(integration, "load_config", lambda a: {**integration.DEFAULTS, "early_stop": False})
    result, _ = await feed(make_wrapped(agent, original), CALL, 9)
    assert result is None

    def boom(a, c):
        raise RuntimeError("boom")

    monkeypatch.setattr(integration, "load_config", lambda a: dict(integration.DEFAULTS))
    monkeypatch.setattr(integration, "applies", boom)
    result, _ = await feed(make_wrapped(agent, original), CALL, 9)
    assert result is None


def test_stream_extension_ignores_calls_without_a_callback(agent):
    data = {"args": (agent,), "kwargs": {"response_callback": None}}
    load(STREAM_EXT).ClaudeToolBridgeStream(agent=agent).execute(data=data)
    assert data["kwargs"]["response_callback"] is None
    load(STREAM_EXT).ClaudeToolBridgeStream(agent=agent).execute(data={"args": (agent,), "kwargs": None})


# ---------------------------------------------------------------- per-turn plan, caches, stop sequence

STOP_ON = {"stop_sequence": True}
STOPPED = CALL[: CALL.index(bridge.STOP_SEQUENCES[0])]  # the call as the provider returns it when it stops
MODEL = SimpleNamespace(
    a0_model_conf=SimpleNamespace(provider="anthropic", name="claude-sonnet-5-5"), kwargs={"max_tokens": 32000}
)


def with_config(monkeypatch, **overrides):
    monkeypatch.setattr(integration, "load_config", lambda a: {**integration.DEFAULTS, **overrides})


async def noop(chunk, full):
    return None


def run_stop_ext(callback, model=MODEL, **kwargs):
    data = {"args": (model,), "kwargs": {"response_callback": callback, **kwargs}}
    load(STOP_EXT).ClaudeToolBridgeStop(agent=None).execute(data=data)
    return data["kwargs"]


ALL_STOPS = [bridge.CALL_STOP, *bridge.LOOP_GUARDS]  # what a request carries: four, the most LiteLLM lets through


def test_the_stop_sequence_the_loop_guard_and_prose_answers_are_on_by_default():
    yaml = (ROOT / "default_config.yaml").read_text(encoding="utf-8")
    for key in ("stop_sequence", "loop_guard", "prose_as_response"):
        assert integration.DEFAULTS[key] is True
        assert f"{key}: true" in yaml


def test_both_extensions_share_one_plan_per_turn(agent, monkeypatch):
    calls = {"config": 0, "applies": 0}

    def config(a):
        calls["config"] += 1
        return dict(integration.DEFAULTS)

    def covered(a, c):
        calls["applies"] += 1
        return True

    monkeypatch.setattr(integration, "load_config", config)
    monkeypatch.setattr(integration, "applies", covered)
    make_wrapped(agent, noop)
    run_loop_ext(agent, "Checking.\n\n" + CALL)
    assert calls == {"config": 1, "applies": 1}


def test_other_models_get_no_wrapper_at_all(agent, monkeypatch):
    monkeypatch.setattr(integration, "applies", lambda a, c: False)
    assert make_wrapped(agent, noop) is noop


def test_stream_extension_publishes_the_stop_request_with_the_callback(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    wrapped = make_wrapped(agent, noop)
    state = agent.loop_data.params_temporary[integration.STATE_KEY]
    assert wrapped.ctb_state is state and state["stop"] is True and state["guard"] is True
    with_config(monkeypatch, stop_sequence=False, loop_guard=False)
    agent.loop_data.params_temporary.clear()
    state = make_wrapped(agent, noop).ctb_state
    assert state["stop"] is False and state["guard"] is False


def test_stop_extension_adds_the_sequences_and_merges_the_ones_already_set(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    wrapped = make_wrapped(agent, noop)
    kwargs = run_stop_ext(wrapped)
    assert kwargs["stop"] == ALL_STOPS and len(ALL_STOPS) == bridge.MAX_STOPS
    assert wrapped.ctb_state["max_tokens"] == 32000
    kwargs = run_stop_ext(wrapped, stop="END")
    assert kwargs["stop"] == ["END", *ALL_STOPS[:3]]  # the model's own strings come first, ours fill the rest
    kwargs = run_stop_ext(wrapped, stop=["END", bridge.CALL_STOP])
    assert kwargs["stop"] == ["END", *ALL_STOPS[:3]]
    model = SimpleNamespace(a0_model_conf=MODEL.a0_model_conf, kwargs={"stop": ["MODEL"], "max_completion_tokens": 9000})
    assert run_stop_ext(wrapped, model=model)["stop"] == ["MODEL", *ALL_STOPS[:3]]
    assert wrapped.ctb_state["max_tokens"] == 9000


def test_stop_extension_does_nothing_unless_asked_or_for_other_models(agent, monkeypatch):
    assert "stop" not in run_stop_ext(noop)  # not a callback of this plugin
    with_config(monkeypatch, stop_sequence=False, loop_guard=False)
    wrapped = make_wrapped(agent, noop)
    assert "stop" not in run_stop_ext(wrapped)
    with_config(monkeypatch, **STOP_ON)
    agent.loop_data.params_temporary.clear()
    wrapped = make_wrapped(agent, noop)
    other = SimpleNamespace(a0_model_conf=SimpleNamespace(provider="openai", name="gpt-5"), kwargs={})
    assert "stop" not in run_stop_ext(wrapped, model=other)
    assert wrapped.ctb_state["stop"] is False and wrapped.ctb_state["guard"] is False
    assert "stop" not in run_stop_ext(wrapped, model=SimpleNamespace(kwargs={}))  # no model config: stay out
    assert run_stop_ext(None) == {"response_callback": None}


def test_loop_extension_completes_and_translates_a_stopped_reply(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    make_wrapped(agent, noop)
    llm, _ = run_loop_ext(agent, "Checking.\n\n" + STOPPED, usage={"completion_tokens": 120})
    request = json.loads(llm.response)
    assert request["tool_name"] == "code_execution_tool" and request["thoughts"] == ["Checking."]
    assert request["tool_args"] == {"runtime": "terminal", "session": 3, "code": "ls\npwd"}
    stock, _ = run_loop_ext(agent, "Checking.\n\n" + CALL)
    assert json.loads(stock.response) == request  # the stop changes the cost, not the call


def test_loop_extension_leaves_a_reply_that_ran_out_of_tokens_alone(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    wrapped = make_wrapped(agent, noop)
    run_stop_ext(wrapped)  # records max_tokens = 32000
    llm, _ = run_loop_ext(agent, STOPPED, usage={"completion_tokens": 32000})
    assert llm.response == STOPPED
    assert not agent.loop_data.params_persistent


def test_loop_extension_does_not_complete_a_reply_nobody_stopped(agent, monkeypatch):
    with_config(monkeypatch, stop_sequence=False, loop_guard=False)
    make_wrapped(agent, noop)
    llm, _ = run_loop_ext(agent, STOPPED, usage={"completion_tokens": 120})
    assert llm.response == STOPPED


def test_a_stopped_reply_without_a_usable_call_is_dropped_and_asked_again_without_a_stop(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    make_wrapped(agent, noop)
    unknown = STOPPED.replace("code_execution_tool", "no_such_tool")
    llm, data = run_loop_ext(agent, unknown, usage={"completion_tokens": 120})
    # an empty reply makes _context_doctor step aside; the flag makes the loop run the turn again
    assert llm.response == "" and data["skip_default_processing"] is True
    assert agent.loop_data.params_persistent[integration.STOP_OFF_KEY] == integration.STOP_PAUSE_TURNS
    for turn in range(integration.STOP_PAUSE_TURNS):  # the retry and the turns after it go without a stop
        agent.loop_data.params_temporary.clear()
        state = make_wrapped(agent, noop).ctb_state
        assert state["stop"] is False and state["guard"] is False, turn
    agent.loop_data.params_temporary.clear()
    state = make_wrapped(agent, noop).ctb_state
    assert state["stop"] is True and state["guard"] is True  # the pause is over


def test_stop_extension_survives_a_broken_state(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    wrapped = make_wrapped(agent, noop)
    wrapped.ctb_state.clear()
    assert "stop" not in run_stop_ext(wrapped)


def test_a_reply_cut_inside_a_call_somewhere_else_is_asked_again_without_a_stop(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    make_wrapped(agent, noop)
    elsewhere = CALL[: CALL.index("</parameter>\n</invoke>")]  # cut inside the call, but not where the stop leaves it
    llm, data = run_loop_ext(agent, elsewhere, usage={"completion_tokens": 120})
    assert llm.response == "" and data["skip_default_processing"] is True
    assert agent.loop_data.params_persistent[integration.STOP_OFF_KEY] == integration.STOP_PAUSE_TURNS


def test_the_stop_stays_on_when_nothing_went_wrong(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    make_wrapped(agent, noop)
    for reply in ("Done, nothing to run.", '{"tool_name": "response", "tool_args": {"text": "hi"}}', STOPPED):
        llm, data = run_loop_ext(agent, reply, usage={"completion_tokens": 120})
        assert "skip_default_processing" not in data and llm.response != ""
    assert not agent.loop_data.params_persistent


# ---- caches (a fake agent with the attributes A0 builds cache keys from)


@pytest.fixture
def scoped():
    integration.reset_caches()
    agent = SimpleNamespace(
        config=SimpleNamespace(profile="dev"),
        context=SimpleNamespace(get_data=lambda key: {"chat_model_override": {"preset_name": "p"}}.get(key)),
        loop_data=SimpleNamespace(params_temporary={}, params_persistent={}),
    )
    yield agent
    integration.reset_caches()


def test_config_and_model_check_are_reused_between_turns_for_a_few_seconds(scoped, monkeypatch):
    reads = {"config": 0, "model": 0}

    def get_plugin_config(name, agent=None, **kwargs):
        reads["config"] += 1
        return {"early_stop": False}

    def covered(agent, config):
        reads["model"] += 1
        return True

    monkeypatch.setattr(integration.plugins, "get_plugin_config", get_plugin_config)
    monkeypatch.setattr(integration, "_model_covered", covered)
    for _ in range(3):
        scoped.loop_data.params_temporary.clear()
        config, applies = integration.turn_plan(scoped)
        assert config["early_stop"] is False and config["stop_sequence"] is True and applies is True
    assert reads == {"config": 1, "model": 1}
    monkeypatch.setattr(integration, "CONFIG_TTL", 0)
    scoped.loop_data.params_temporary.clear()
    integration.turn_plan(scoped)
    assert reads == {"config": 2, "model": 2}


def test_model_check_is_not_shared_between_chat_model_overrides(scoped, monkeypatch):
    answers = iter([True, False])
    monkeypatch.setattr(integration, "_model_covered", lambda a, c: next(answers))
    assert integration.applies(scoped, integration.DEFAULTS) is True
    scoped.context = SimpleNamespace(get_data=lambda key: {"preset_name": "other"})
    assert integration.applies(scoped, integration.DEFAULTS) is False


def test_registry_is_built_once_per_profile_and_project(scoped, monkeypatch):
    builds = []
    monkeypatch.setattr(integration, "_build_registry", lambda a: builds.append(1) or bridge.build_registry(TOOLS))
    first = integration.get_registry(scoped)
    scoped.loop_data.params_temporary.clear()
    assert integration.get_registry(scoped) is first and len(builds) == 1
    assert scoped.loop_data.params_temporary[integration.REGISTRY_CACHED_KEY] is True


def test_a_tool_added_after_the_registry_was_cached_is_picked_up_once(scoped, monkeypatch):
    older = bridge.build_registry([TOOLS[0]])
    newer = bridge.build_registry([TOOLS[0], {"name": "fresh_tool", "description": "", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}}}])
    registries = iter([older, newer, newer])
    monkeypatch.setattr(integration, "_build_registry", lambda a: next(registries))
    call = '<invoke name="fresh_tool">\n<parameter name="q">hi</parameter>\n</invoke>'
    assert integration.translate_response(scoped, CALL) is not None  # builds and caches `older`
    scoped.loop_data.params_temporary.clear()
    request, _ = integration.translate_response(scoped, call)  # cached copy does not know it: rebuilt once
    assert request == {"tool_name": "fresh_tool", "tool_args": {"q": "hi"}}
    scoped.loop_data.params_temporary.clear()
    assert integration.translate_response(scoped, call) is not None  # the new registry is the cached one now
    scoped.loop_data.params_temporary.clear()
    assert integration.translate_response(scoped, '<invoke name="ghost">\n<parameter name="q">x</parameter>\n</invoke>') is None


def test_settings_screen_covers_every_setting():
    import re

    html = (ROOT / "webui" / "config.html").read_text(encoding="utf-8")
    keys = re.findall(r"^(\w+):", (ROOT / "default_config.yaml").read_text(encoding="utf-8"), flags=re.M)
    assert keys and set(keys) == set(integration.DEFAULTS)
    for key in keys:
        assert f"config.{key}" in html, key
    assert 'settings_sections: ["agent"]' in (ROOT / "plugin.yaml").read_text(encoding="utf-8")


def test_a_stopped_reply_with_loose_parameters_and_no_opening_tag_is_asked_again(agent, monkeypatch):
    with_config(monkeypatch, **STOP_ON)
    make_wrapped(agent, noop)
    loose = '<parameter name="thoughts">["thinking"]</paramete'  # the stop fired before any <invoke name=...> existed
    llm, data = run_loop_ext(agent, loose, usage={"completion_tokens": 40})
    assert llm.response == "" and data["skip_default_processing"] is True
    assert agent.loop_data.params_persistent[integration.STOP_OFF_KEY] == integration.STOP_PAUSE_TURNS
    # without a stop requested, the same text is left alone
    agent2 = SimpleNamespace(loop_data=SimpleNamespace(params_temporary={}, params_persistent={}))
    llm, data = run_loop_ext(agent2, loose)
    assert llm.response == loose and "skip_default_processing" not in data


# ---------------------------------------------------------------- loop guards, silent retries, prose answers

EMPTY_LOOP = "\n" + '<invoke name="code_execution_tool">\n</invoke>\n\n\n' * 300
CUT_LOOP = EMPTY_LOOP[: EMPTY_LOOP.index("</invoke>")]  # what the provider returns when it stops at the guard
RESPONSE_TOOL = {"name": "response", "description": "", "parameters": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
JSON_CALL = '{"tool_name": "response", "tool_args": {"text": "hi"}}'


@pytest.fixture
def answering(agent, monkeypatch):
    """An agent whose tools include `response`, as in Agent Zero."""
    registry = bridge.build_registry([*TOOLS, RESPONSE_TOOL])
    monkeypatch.setattr(integration, "get_registry", lambda a: registry)
    return agent


def turn(agent, monkeypatch=None, **config):
    """Start a turn the way the stream extension does (so the state carries the switches) and record max_tokens."""
    if monkeypatch is not None:
        with_config(monkeypatch, **config)
    agent.loop_data.params_temporary.clear()
    wrapped = make_wrapped(agent, noop)
    run_stop_ext(wrapped)
    return wrapped


def test_a_loop_cut_at_its_second_unit_is_asked_again_with_the_guards_still_on(agent, monkeypatch):
    turn(agent, monkeypatch)
    llm, data = run_loop_ext(agent, CUT_LOOP, usage={"completion_tokens": 30})
    assert llm.response == "" and data["skip_default_processing"] is True
    persistent = agent.loop_data.params_persistent
    assert persistent[integration.RETRY_KEY] == 1 and integration.STOP_OFF_KEY not in persistent  # no pause
    state = turn(agent).ctb_state
    assert state["guard"] is True and state["stop"] is True


def test_only_a_few_replies_are_thrown_away_in_a_row_and_a_usable_reply_ends_the_spell(agent, monkeypatch):
    turn(agent, monkeypatch)
    for _ in range(integration.SILENT_RETRY_LIMIT):
        llm, data = run_loop_ext(agent, CUT_LOOP, usage={"completion_tokens": 30})
        assert llm.response == "" and data["skip_default_processing"] is True
    llm, data = run_loop_ext(agent, CUT_LOOP, usage={"completion_tokens": 30})  # spent: the core gets it
    assert llm.response == CUT_LOOP and "skip_default_processing" not in data
    assert integration.retry_pending(agent)
    run_loop_ext(agent, JSON_CALL)  # the model recovered
    assert not integration.retry_pending(agent)
    llm, data = run_loop_ext(agent, CUT_LOOP, usage={"completion_tokens": 30})
    assert llm.response == "" and data["skip_default_processing"] is True


def test_a_reply_that_ran_to_max_tokens_as_a_loop_is_thrown_away_too(agent, monkeypatch):
    turn(agent, monkeypatch)
    llm, data = run_loop_ext(agent, EMPTY_LOOP, usage={"completion_tokens": 32000})
    assert llm.response == "" and data["skip_default_processing"] is True


def test_the_loop_guard_can_be_turned_off(agent, monkeypatch):
    turn(agent, monkeypatch, stop_sequence=False, loop_guard=False)
    for reply in (CUT_LOOP, EMPTY_LOOP):
        llm, data = run_loop_ext(agent, reply, usage={"completion_tokens": 32000})
        assert llm.response == reply and "skip_default_processing" not in data
    assert not agent.loop_data.params_persistent


def test_a_repetitive_call_is_not_a_loop(agent, monkeypatch):
    turn(agent, monkeypatch)
    call = json.dumps({"tool_name": "code_execution_tool", "tool_args": {"code": "invoke it\n" * 400}})
    llm, data = run_loop_ext(agent, call, usage={"completion_tokens": 900})
    assert llm.response == call and "skip_default_processing" not in data
    mixed = EMPTY_LOOP[:3000] + "\n" + call  # empty calls and then a JSON call: the core's repair takes it
    llm, data = run_loop_ext(agent, mixed, usage={"completion_tokens": 900})
    assert llm.response == mixed and "skip_default_processing" not in data


def test_a_call_cut_inside_a_value_by_a_guard_is_asked_again_with_everything_paused(agent, monkeypatch):
    turn(agent, monkeypatch, stop_sequence=False)  # only the guards are in the request
    elsewhere = CALL[: CALL.index("ls")]
    llm, data = run_loop_ext(agent, elsewhere, usage={"completion_tokens": 40})
    assert llm.response == "" and data["skip_default_processing"] is True
    assert agent.loop_data.params_persistent[integration.STOP_OFF_KEY] == integration.STOP_PAUSE_TURNS
    state = turn(agent).ctb_state
    assert state["stop"] is False and state["guard"] is False


def test_a_stopped_reply_and_a_loop_share_the_retry_budget(agent, monkeypatch):
    unknown = STOPPED.replace("code_execution_tool", "no_such_tool")
    for reply in (unknown, CUT_LOOP):
        agent.loop_data.params_persistent.pop(integration.STOP_OFF_KEY, None)
        turn(agent, monkeypatch)
        llm, _ = run_loop_ext(agent, reply, usage={"completion_tokens": 30})
        assert llm.response == ""
    agent.loop_data.params_persistent.pop(integration.STOP_OFF_KEY, None)
    turn(agent, monkeypatch)
    llm, data = run_loop_ext(agent, unknown, usage={"completion_tokens": 30})
    assert llm.response == unknown and "skip_default_processing" not in data  # budget spent
    assert agent.loop_data.params_persistent[integration.STOP_OFF_KEY] == integration.STOP_PAUSE_TURNS  # still paused


def test_without_loop_data_there_is_nothing_to_remember_so_nothing_is_thrown_away():
    assert integration.allow_silent_retry(SimpleNamespace()) is False
    assert integration.allow_silent_retry(SimpleNamespace(loop_data=SimpleNamespace(params_persistent=None))) is False


def test_switches_are_off_while_a_pause_runs_and_the_pause_counts_down_once_per_turn(agent):
    config = dict(integration.DEFAULTS)
    assert integration.turn_switches(agent, config) == (True, True)
    integration.stop_failed(agent)
    assert [integration.turn_switches(agent, config) for _ in range(integration.STOP_PAUSE_TURNS)] == [(False, False)] * 3
    assert integration.turn_switches(agent, config) == (True, True)
    assert integration.turn_switches(agent, {**config, "stop_sequence": False}) == (False, True)
    assert integration.turn_switches(agent, {**config, "loop_guard": False}) == (True, False)


def test_the_stop_extension_sends_only_the_sequences_that_are_switched_on(agent, monkeypatch):
    turn(agent, monkeypatch, stop_sequence=False)
    assert run_stop_ext(make_wrapped(agent, noop))["stop"] == bridge.LOOP_GUARDS
    agent.loop_data.params_temporary.clear()
    with_config(monkeypatch, loop_guard=False)
    assert run_stop_ext(make_wrapped(agent, noop))["stop"] == [bridge.CALL_STOP]


def test_what_does_not_fit_in_the_four_stop_strings_is_not_expected_back(agent, monkeypatch):
    with_config(monkeypatch)
    for existing, stop, guard in [(["A", "B", "C", "D"], False, False), (["A", "B", "C"], True, False), (["A"], True, True)]:
        agent.loop_data.params_temporary.clear()
        wrapped = make_wrapped(agent, noop)
        kwargs = run_stop_ext(wrapped, stop=existing)
        assert kwargs["stop"] == bridge.stops_to_send(existing, True, True) and len(kwargs["stop"]) <= bridge.MAX_STOPS
        assert (wrapped.ctb_state["stop"], wrapped.ctb_state["guard"]) == (stop, guard), existing


# ---- prose

PROSE = (
    "The query returns 3 rows.\n\n"
    + " ".join(f"word{i} sentence {i * 7919 % 1013} about topic" for i in range(40))
    + "\n\nThe third one is the problem: it has no key."
)
assert bridge.PROSE_MIN_CHARS <= len(PROSE) < bridge.DEGENERATE_MIN


def test_a_prose_reply_goes_out_through_the_response_tool(answering, monkeypatch):
    with_config(monkeypatch)
    llm, data = run_loop_ext(answering, "  " + PROSE + "\n")
    assert json.loads(llm.response) == {"tool_name": "response", "tool_args": {"text": PROSE}}
    assert "skip_default_processing" not in data
    transformed = transform_response(llm.response, suppress_xml=True, split_thoughts=True)
    assert looks_like_tool_call(llm.response, transformed)
    name, args = extract_tools.normalize_tool_request(extract_tools.extract_tool_request(transformed))
    assert name == "response" and args["text"] == PROSE


def test_prose_with_a_quoted_example_is_still_an_answer(answering, monkeypatch):
    with_config(monkeypatch)
    text = "This is the format:\n```xml\n" + CALL + "\n```\nUse it like that. " + PROSE
    llm, _ = run_loop_ext(answering, text)
    assert json.loads(llm.response)["tool_args"]["text"] == text


@pytest.mark.parametrize(
    "reply",
    [
        "Let me check the tables first:",  # announces the next step
        "Short answer: yes.",  # a short reply is most often a sentence before the next call, not the answer
        PROSE + ":",  # a long one that ends on a colon announces something too
        "I will run it.\n" + CALL.replace("code_execution_tool", "no_such_tool"),  # a call to nothing
        'Here it is:\n```json\n{"tool_name": "response", "tool_args": {"text": "x"}}\n```',
        "Done <function_calls>",
        "x " * 1000,  # a loop of words
    ],
)
def test_prose_is_not_turned_into_an_answer_when_it_is_not_one(answering, monkeypatch, reply):
    with_config(monkeypatch)
    llm, data = run_loop_ext(answering, reply)
    assert llm.response == reply and "skip_default_processing" not in data


def test_prose_stays_as_it_is_when_asked_to_or_cut_short_or_without_a_response_tool(answering, agent, monkeypatch):
    with_config(monkeypatch, prose_as_response=False)
    assert run_loop_ext(answering, PROSE)[0].response == PROSE
    turn(answering, monkeypatch)
    assert run_loop_ext(answering, PROSE, usage={"completion_tokens": 32000})[0].response == PROSE  # cut by the limit
    registry = bridge.build_registry(TOOLS)  # no `response` tool known
    monkeypatch.setattr(integration, "get_registry", lambda a: registry)
    with_config(monkeypatch)
    assert run_loop_ext(agent, PROSE)[0].response == PROSE


def test_a_prose_answer_ends_the_spell_of_thrown_away_replies(answering, monkeypatch):
    turn(answering, monkeypatch)
    run_loop_ext(answering, CUT_LOOP, usage={"completion_tokens": 30})
    assert integration.retry_pending(answering)
    llm, _ = run_loop_ext(answering, PROSE, usage={"completion_tokens": 30})
    assert json.loads(llm.response)["tool_name"] == "response" and not integration.retry_pending(answering)
