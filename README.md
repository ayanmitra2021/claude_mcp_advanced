# MCP Logging and Progress Demo

## Setup

Ensure you've set the `ANTHROPIC_API_KEY` env variable with a valid key.

Install dependencies using uv:

```bash
uv sync
```

## Running the Project

Run the MCP server:
```bash
uv run server.py
```

## Run the MCP client:

```bash
uv run client.py
```

## Run the MCP inspector

```bash
npx @modelcontextprotocol/inspector python server.py
```