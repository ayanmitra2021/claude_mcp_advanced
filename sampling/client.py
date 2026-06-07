import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from aioconsole import ainput
from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    LoggingMessageNotificationParams,
    ProgressNotification,
    TextContent,
    SamplingMessage,
)

anthropic_client = AsyncAnthropic()
model = "claude-haiku-4-5"

server_params = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
)


async def call_claude(
    messages: list[dict],
    system: str | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
):
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    return await anthropic_client.messages.create(**kwargs)


async def sampling_callback(
    context: RequestContext, params: CreateMessageRequestParams
) -> CreateMessageResult:
    messages = [
        {"role": msg.role, "content": msg.content.text}
        for msg in params.messages
        if msg.content.type == "text"
    ]
    system = getattr(params, "systemPrompt", None) or getattr(params, "system_prompt", None)
    response = await call_claude(messages, system=system)
    text = "".join(p.text for p in response.content if p.type == "text")
    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )


async def agent_loop(session: ClientSession, user_message: str) -> str:
    tools_response = await session.list_tools()
    tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools_response.tools
    ]

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = await call_claude(messages, tools=tools)

        if response.stop_reason == "end_turn":
            return "".join(p.text for p in response.content if p.type == "text")

        if response.stop_reason == "tool_use":
            tool_block = next((p for p in response.content if p.type == "tool_use"), None)
            if not tool_block:
                return "".join(p.text for p in response.content if p.type == "text")

            messages.append({"role": "assistant", "content": response.content})

            print(f"\n[Calling tool: {tool_block.name} with {tool_block.input}]")

            tool_result = await session.call_tool(
                name=tool_block.name,
                arguments=tool_block.input,
                progress_callback=print_progress_callback,
            )

            result_text = "".join(
                item.text for item in tool_result.content if hasattr(item, "text")
            )

            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_text,
                }],
            })
        else:
            return "".join(p.text for p in response.content if p.type == "text")

async def logging_callback(params: LoggingMessageNotificationParams):
    print(f"\n[Server] {params.data}")


async def message_handler(message) -> None:
    # message_handler receives ServerNotification objects; unwrap RootModel if needed
    notification = getattr(message, "root", message)
    if isinstance(notification, ProgressNotification):
        p = notification.params
        await print_progress_callback(p.progress, p.total, None)


async def print_progress_callback(
    progress: float, total: float | None, message: str | None
):
    if total is not None:
        percentage = (progress / total) * 100
        filled = int(percentage // 10)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"\r  [{bar}] {percentage:.0f}%  ", end="", flush=True)
        if progress >= total:
            print()
    else:
        print(f"\r  Progress: {progress:.0f}", end="", flush=True)

async def run():
    print("Research Assistant")
    print("------------------")
    print("Ask me to research any topic. Type 'exit' to quit.\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write,
            sampling_callback=sampling_callback,
            logging_callback=logging_callback,
            message_handler=message_handler,
        ) as session:
            await session.initialize()

            while True:
                try:
                    user_input = await ainput("You: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                response = await agent_loop(session, user_input)
                print(f"\nAssistant: {response}\n")


def _suppress_event_loop_closed_on_exit(exc_info):
    # Windows asyncio bug: subprocess transports try to close during GC after
    # the event loop is already gone. Safe to ignore.
    if exc_info.exc_type is RuntimeError and "Event loop is closed" in str(exc_info.exc_value):
        return
    sys.__unraisablehook__(exc_info)


if __name__ == "__main__":
    sys.unraisablehook = _suppress_event_loop_closed_on_exit
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
