from unittest.mock import AsyncMock, MagicMock, patch

import client as client_module
from client import (
    agent_loop,
    call_claude,
    logging_callback,
    message_handler,
    print_progress_callback,
    sampling_callback,
)
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    LoggingMessageNotificationParams,
    ProgressNotification,
    ProgressNotificationParams,
    SamplingMessage,
    TextContent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_end_turn_response(text: str = "answer"):
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    return response


def _make_tool_use_response(name: str, input: dict, tool_id: str = "tool_abc"):
    response = MagicMock()
    response.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input
    block.id = tool_id
    response.content = [block]
    return response


def _make_tool_result(text: str = "tool output"):
    result = MagicMock()
    item = MagicMock()
    item.text = text
    result.content = [item]
    return result


def _make_session(tool_names: list[str] | None = None):
    session = AsyncMock()
    tools = []
    for name in (tool_names or ["test_tool"]):
        t = MagicMock()
        t.name = name
        t.description = f"Description for {name}"
        t.inputSchema = {"type": "object", "properties": {}}
        tools.append(t)
    session.list_tools.return_value = MagicMock(tools=tools)
    session.call_tool.return_value = _make_tool_result()
    return session


# ---------------------------------------------------------------------------
# print_progress_callback
# ---------------------------------------------------------------------------

async def test_print_progress_callback_with_total_shows_percentage(capsys):
    await print_progress_callback(40.0, 100.0, None)
    assert "40" in capsys.readouterr().out


async def test_print_progress_callback_without_total_shows_value(capsys):
    await print_progress_callback(7.0, None, None)
    assert "7" in capsys.readouterr().out


async def test_print_progress_callback_at_100_ends_with_newline(capsys):
    await print_progress_callback(100.0, 100.0, None)
    assert capsys.readouterr().out.endswith("\n")


async def test_print_progress_callback_partial_does_not_end_with_newline(capsys):
    await print_progress_callback(50.0, 100.0, None)
    # In-progress updates use end="" so no trailing newline until complete
    assert not capsys.readouterr().out.endswith("\n")


# ---------------------------------------------------------------------------
# logging_callback
# ---------------------------------------------------------------------------

async def test_logging_callback_prints_server_message(capsys):
    params = LoggingMessageNotificationParams(level="info", data="Processing data")
    await logging_callback(params)
    out = capsys.readouterr().out
    assert "Processing data" in out
    assert "[Server]" in out


# ---------------------------------------------------------------------------
# message_handler
# ---------------------------------------------------------------------------

async def test_message_handler_delegates_progress_to_callback():
    notification = ProgressNotification(
        method="notifications/progress",
        params=ProgressNotificationParams(progressToken="tok1", progress=50.0, total=100.0),
    )
    message = MagicMock()
    message.root = notification

    with patch.object(client_module, "print_progress_callback", new=AsyncMock()) as mock_cb:
        await message_handler(message)
        mock_cb.assert_called_once_with(50.0, 100.0, None)


async def test_message_handler_ignores_non_progress_notifications():
    message = MagicMock()
    message.root = MagicMock()  # not a ProgressNotification
    await message_handler(message)  # must not raise


async def test_message_handler_unwraps_root_attribute():
    """Notifications arrive wrapped in a RootModel; message_handler must unwrap them."""
    notification = ProgressNotification(
        method="notifications/progress",
        params=ProgressNotificationParams(progressToken="tok2", progress=10.0, total=100.0),
    )
    # Simulate the RootModel wrapper the MCP SDK produces
    wrapper = MagicMock()
    wrapper.root = notification

    with patch.object(client_module, "print_progress_callback", new=AsyncMock()) as mock_cb:
        await message_handler(wrapper)
        mock_cb.assert_called_once_with(10.0, 100.0, None)


# ---------------------------------------------------------------------------
# call_claude
# ---------------------------------------------------------------------------

async def test_call_claude_sends_model_and_messages():
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude(messages)
        kwargs = mock_ac.messages.create.call_args.kwargs
        assert kwargs["model"] == client_module.model
        assert kwargs["messages"] == messages
        assert kwargs["max_tokens"] == 4000


async def test_call_claude_includes_system_when_given():
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude([], system="Be concise")
        assert mock_ac.messages.create.call_args.kwargs["system"] == "Be concise"


async def test_call_claude_omits_system_when_not_given():
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude([])
        assert "system" not in mock_ac.messages.create.call_args.kwargs


async def test_call_claude_includes_tools_when_given():
    tools = [{"name": "search", "description": "Search", "input_schema": {}}]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude([], tools=tools)
        assert mock_ac.messages.create.call_args.kwargs["tools"] == tools


async def test_call_claude_omits_tools_when_not_given():
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude([])
        assert "tools" not in mock_ac.messages.create.call_args.kwargs


async def test_call_claude_respects_custom_max_tokens():
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=MagicMock())
        await call_claude([], max_tokens=512)
        assert mock_ac.messages.create.call_args.kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# sampling_callback
# ---------------------------------------------------------------------------

async def test_sampling_callback_returns_create_message_result():
    params = CreateMessageRequestParams(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hello"))],
        maxTokens=100,
    )
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="world")]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=mock_response)
        result = await sampling_callback(MagicMock(), params)
    assert isinstance(result, CreateMessageResult)
    assert result.content.text == "world"


async def test_sampling_callback_converts_text_messages_to_anthropic_format():
    params = CreateMessageRequestParams(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="research this"))],
        maxTokens=100,
    )
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="summary")]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=mock_response)
        await sampling_callback(MagicMock(), params)
        sent_messages = mock_ac.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": "research this"}]


async def test_sampling_callback_passes_system_prompt():
    params = CreateMessageRequestParams(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
        maxTokens=100,
        systemPrompt="You are an expert",
    )
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="ok")]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=mock_response)
        await sampling_callback(MagicMock(), params)
        assert mock_ac.messages.create.call_args.kwargs["system"] == "You are an expert"


async def test_sampling_callback_filters_non_text_messages():
    """Non-text SamplingMessages should be silently skipped."""
    image_msg = MagicMock()
    image_msg.content = MagicMock(type="image")
    text_msg = SamplingMessage(role="user", content=TextContent(type="text", text="hello"))

    params = MagicMock()
    params.messages = [image_msg, text_msg]
    params.systemPrompt = None
    params.system_prompt = None

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="ok")]
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=mock_response)
        await sampling_callback(MagicMock(), params)
        sent_messages = mock_ac.messages.create.call_args.kwargs["messages"]
    assert len(sent_messages) == 1
    assert sent_messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# agent_loop
# ---------------------------------------------------------------------------

async def test_agent_loop_returns_text_on_end_turn():
    session = _make_session()
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=_make_end_turn_response("The answer is 42"))
        result = await agent_loop(session, "what is the answer?")
    assert result == "The answer is 42"


async def test_agent_loop_calls_tool_on_tool_use():
    session = _make_session(["research_topic"])
    session.call_tool.return_value = _make_tool_result("research output")

    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(side_effect=[
            _make_tool_use_response("research_topic", {"topic": "mars"}),
            _make_end_turn_response("final answer"),
        ])
        result = await agent_loop(session, "tell me about mars")

    assert result == "final answer"
    session.call_tool.assert_called_once()
    assert session.call_tool.call_args.kwargs["name"] == "research_topic"


async def test_agent_loop_passes_progress_callback_to_call_tool():
    session = _make_session()
    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(side_effect=[
            _make_tool_use_response("test_tool", {}),
            _make_end_turn_response("done"),
        ])
        await agent_loop(session, "do something")

    assert session.call_tool.call_args.kwargs.get("progress_callback") is client_module.print_progress_callback


async def test_agent_loop_handles_missing_tool_block():
    """stop_reason=tool_use but no tool_use block in content → return empty text, no tool call."""
    session = _make_session()
    response = MagicMock()
    response.stop_reason = "tool_use"
    text_only = MagicMock()
    text_only.type = "text"
    text_only.text = ""
    response.content = [text_only]

    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=response)
        result = await agent_loop(session, "do something")

    session.call_tool.assert_not_called()
    assert result == ""


async def test_agent_loop_appends_tool_result_for_next_call():
    session = _make_session(["lookup"])
    session.call_tool.return_value = _make_tool_result("lookup result")

    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(side_effect=[
            _make_tool_use_response("lookup", {}, tool_id="t1"),
            _make_end_turn_response("done"),
        ])
        await agent_loop(session, "look something up")

    second_call_messages = mock_ac.messages.create.call_args_list[1].kwargs["messages"]
    last_msg = second_call_messages[-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["type"] == "tool_result"
    assert last_msg["content"][0]["content"] == "lookup result"
    assert last_msg["content"][0]["tool_use_id"] == "t1"


async def test_agent_loop_returns_text_on_unexpected_stop_reason():
    session = _make_session()
    response = MagicMock()
    response.stop_reason = "max_tokens"
    block = MagicMock()
    block.type = "text"
    block.text = "truncated"
    response.content = [block]

    with patch.object(client_module, "anthropic_client") as mock_ac:
        mock_ac.messages.create = AsyncMock(return_value=response)
        result = await agent_loop(session, "a question")

    assert result == "truncated"
