from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent
import httpx
from bs4 import BeautifulSoup
from pydantic import Field

mcp = FastMCP(name="Demo Server")

# Wikimedia policy requires a UA with a contact email or URL; without it all domains return 403
HEADERS = {"User-Agent": "research-demo/1.0 (educational; 2006.ayan@gmail.com)"}
# Scholarpedia resets connections from bot UAs; use a browser-like string
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

WIKI_API = "https://en.wikipedia.org/w/api.php"
SIMPLE_WIKI_API = "https://simple.wikipedia.org/w/api.php"

_WIKI_PARAMS = {
    "action": "query",
    "prop": "extracts",
    "exintro": "1",
    "explaintext": "1",
    "format": "json",
    "redirects": "1",
}


def _make_slug(topic: str) -> str:
    slug = topic.strip().replace(" ", "_")
    return slug[0].upper() + slug[1:] if slug else slug


async def _query_mediawiki(api_url: str, title: str) -> str:
    params = {**_WIKI_PARAMS, "titles": title}
    async with httpx.AsyncClient(headers=HEADERS) as client:
        response = await client.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return f"No article found for '{title}'"
        return (page.get("extract") or "")[:500]


async def fetch_english_wikipedia(topic: str) -> str:
    try:
        return await _query_mediawiki(WIKI_API, _make_slug(topic))
    except Exception as e:
        return f"Error fetching English Wikipedia ({type(e).__name__}): {e}"


async def fetch_simple_wikipedia(topic: str) -> str:
    try:
        return await _query_mediawiki(SIMPLE_WIKI_API, _make_slug(topic))
    except Exception as e:
        return f"Error fetching Simple Wikipedia ({type(e).__name__}): {e}"


async def fetch_scholarpedia(topic: str) -> str:
    slug = _make_slug(topic)
    url = f"https://www.scholarpedia.org/article/{slug}"
    try:
        async with httpx.AsyncClient(headers=BROWSER_HEADERS, follow_redirects=True) as client:
            response = await client.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:500]
    except Exception as e:
        return ""


async def research(topic_name: str, ctx: Context) -> list[str]:
    await ctx.info("Please wait while I am extracting the data")
    await ctx.report_progress(10, 100)

    results = []
    results.append(await fetch_english_wikipedia(topic_name))
    await ctx.report_progress(40, 100)

    results.append(await fetch_simple_wikipedia(topic_name))
    await ctx.report_progress(70, 100)

    results.append(await fetch_scholarpedia(topic_name))
    await ctx.report_progress(100, 100)

    return results


async def summarize_at_client(text_to_summarize: str, ctx: Context):
    prompt = f"""
        Please summarize the following text:
        {text_to_summarize}
    """

    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user", content=TextContent(type="text", text=prompt)
            )
        ],
        max_tokens=4000,
        system_prompt="You are a helpful research assistant.",
    )

    if result.content.type == "text":
        return result.content.text
    else:
        raise ValueError("Sampling failed")


@mcp.tool(
    name="research_topic",
    description="This tool takes a topic as an input and researches about it in 3 credible sources"
)
async def research_topic(
    topic_to_research: str = Field(description="The topic to be researched"),
    ctx: Context = Field(description="The context that is used for sampling")
) -> str:
    research_results = await research(topic_to_research, ctx)
    combined_result = ";".join(research_results)
    await ctx.info("Please wait while I am summarizing the extracted data")
    summary = await summarize_at_client(combined_result, ctx)
    await ctx.info("Summarization complete")
    return summary


if __name__ == "__main__":
    mcp.run(transport="stdio")
