# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `sampling/`:

```bash
uv sync          # Install / sync dependencies
uv run client.py # Run the demo (client spawns the server internally)
```

There is no test suite or linter configured.

## Environment

`ANTHROPIC_API_KEY` must be set (via `.env` at the repo root or shell environment) before running. The client reads it automatically through the Anthropic SDK's default env-var lookup.

## Architecture

This repo demonstrates the **MCP sampling pattern**: a mechanism that lets an MCP server delegate LLM inference back to the client rather than calling an LLM directly.

### The sampling loop

```
client.py  ──calls──>  summarize tool (server.py)
                              │
                    ctx.session.create_message()
                              │
           <──sampling_callback──  client.py
                              │
                    anthropic_client.messages.create()
                              │
           ──CreateMessageResult──>  server.py
                              │
           <──tool result──────────  client.py
```

1. `client.py` connects to `server.py` as a subprocess over stdio (`StdioServerParameters`).
2. `ClientSession` is initialized with a `sampling_callback` — this is the hook the server uses to reach back to the client for LLM calls.
3. The client calls the `summarize` tool on the server.
4. Inside `summarize`, the server calls `ctx.session.create_message(...)`, which fires the client's `sampling_callback`.
5. `sampling_callback` translates `CreateMessageRequestParams` → Anthropic SDK call → `CreateMessageResult` and returns it to the server.
6. The server returns the summarized text as the tool result.

### Key design point

The server (`server.py`) has **no API key and makes no direct LLM calls**. All model inference is owned by the client. This separation means the server stays credential-free and the client controls which model, rate limits, and billing account are used.

### Files

| File | Role |
|------|------|
| `sampling/server.py` | FastMCP server; exposes one tool (`summarize`) that uses sampling |
| `sampling/client.py` | MCP client; owns the Anthropic SDK client and implements `sampling_callback` |
| `sampling/pyproject.toml` | Dependencies: `mcp[cli]`, `anthropic`, `aioconsole`; Python ≥ 3.10 |

### Model in use

`client.py` hard-codes `claude-haiku-4-5`. Change the `model` variable at the top of that file to swap models.
