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
    agent = SimpleNamespace(loop_data=SimpleNamespace(params_temporary={}))
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
