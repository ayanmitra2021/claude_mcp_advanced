import httpx
import pytest
import respx

from server import (
    SIMPLE_WIKI_API,
    WIKI_API,
    _make_slug,
    fetch_english_wikipedia,
    fetch_scholarpedia,
    fetch_simple_wikipedia,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wiki_ok(extract: str) -> dict:
    return {"query": {"pages": {"1": {"pageid": 1, "title": "Test", "extract": extract}}}}

def _wiki_missing() -> dict:
    return {"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}

_SCHOLARPEDIA_HTML = """
<html><body>
  <nav>Navigation links</nav>
  <header>Site header</header>
  <p>Black holes are regions of spacetime where gravity is so strong that nothing can escape.</p>
  <footer>Site footer</footer>
</body></html>
"""

# ---------------------------------------------------------------------------
# _make_slug
# ---------------------------------------------------------------------------

def test_make_slug_capitalises_first_letter():
    assert _make_slug("black holes") == "Black_holes"

def test_make_slug_preserves_existing_case():
    assert _make_slug("Black_hole") == "Black_hole"

def test_make_slug_strips_surrounding_whitespace():
    assert _make_slug("  quantum mechanics  ") == "Quantum_mechanics"

def test_make_slug_empty_string_returns_empty():
    assert _make_slug("") == ""

# ---------------------------------------------------------------------------
# fetch_english_wikipedia
# ---------------------------------------------------------------------------

@respx.mock
async def test_fetch_english_wikipedia_success():
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_ok("A black hole is a region of spacetime.")))
    result = await fetch_english_wikipedia("black holes")
    assert "black hole" in result.lower()

@respx.mock
async def test_fetch_english_wikipedia_missing_article():
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_missing()))
    result = await fetch_english_wikipedia("xyzzy nonexistent")
    assert "No article found" in result

@respx.mock
async def test_fetch_english_wikipedia_http_error_returns_error_string():
    respx.get(WIKI_API).mock(return_value=httpx.Response(403))
    result = await fetch_english_wikipedia("black holes")
    assert "Error" in result
    assert "HTTPStatusError" in result

@respx.mock
async def test_fetch_english_wikipedia_truncates_at_500_chars():
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_ok("x" * 1000)))
    result = await fetch_english_wikipedia("black holes")
    assert len(result) == 500

@respx.mock
async def test_fetch_english_wikipedia_empty_extract_returns_empty_string():
    respx.get(WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_ok("")))
    result = await fetch_english_wikipedia("black holes")
    assert result == ""

# ---------------------------------------------------------------------------
# fetch_simple_wikipedia
# ---------------------------------------------------------------------------

@respx.mock
async def test_fetch_simple_wikipedia_success():
    respx.get(SIMPLE_WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_ok("A black hole is a very dense object.")))
    result = await fetch_simple_wikipedia("black holes")
    assert "black hole" in result.lower()

@respx.mock
async def test_fetch_simple_wikipedia_missing_article():
    respx.get(SIMPLE_WIKI_API).mock(return_value=httpx.Response(200, json=_wiki_missing()))
    result = await fetch_simple_wikipedia("xyzzy nonexistent")
    assert "No article found" in result

@respx.mock
async def test_fetch_simple_wikipedia_connection_error_returns_error_string():
    respx.get(SIMPLE_WIKI_API).mock(side_effect=httpx.ConnectError("Connection refused"))
    result = await fetch_simple_wikipedia("black holes")
    assert "Error" in result
    assert "ConnectError" in result

@respx.mock
async def test_fetch_simple_wikipedia_http_error_returns_error_string():
    respx.get(SIMPLE_WIKI_API).mock(return_value=httpx.Response(500))
    result = await fetch_simple_wikipedia("black holes")
    assert "Error" in result
    assert "HTTPStatusError" in result

# ---------------------------------------------------------------------------
# fetch_scholarpedia
# ---------------------------------------------------------------------------

@respx.mock
async def test_fetch_scholarpedia_success_parses_body_text():
    respx.get("https://www.scholarpedia.org/article/Black_holes").mock(
        return_value=httpx.Response(200, text=_SCHOLARPEDIA_HTML)
    )
    result = await fetch_scholarpedia("black holes")
    assert "Black holes" in result

@respx.mock
async def test_fetch_scholarpedia_strips_nav_header_footer():
    respx.get("https://www.scholarpedia.org/article/Black_holes").mock(
        return_value=httpx.Response(200, text=_SCHOLARPEDIA_HTML)
    )
    result = await fetch_scholarpedia("black holes")
    assert "Navigation links" not in result
    assert "Site header" not in result
    assert "Site footer" not in result

@respx.mock
async def test_fetch_scholarpedia_http_error_returns_error_string():
    respx.get("https://www.scholarpedia.org/article/Black_holes").mock(
        return_value=httpx.Response(404)
    )
    result = await fetch_scholarpedia("black holes")
    assert "Error" in result
    assert "HTTPStatusError" in result

@respx.mock
async def test_fetch_scholarpedia_connection_error_includes_exception_type():
    respx.get("https://www.scholarpedia.org/article/Black_holes").mock(
        side_effect=httpx.ConnectError("")
    )
    result = await fetch_scholarpedia("black holes")
    assert "Error" in result
    assert "ConnectError" in result

@respx.mock
async def test_fetch_scholarpedia_truncates_at_500_chars():
    long_html = f"<html><body><p>{'word ' * 300}</p></body></html>"
    respx.get("https://www.scholarpedia.org/article/Black_holes").mock(
        return_value=httpx.Response(200, text=long_html)
    )
    result = await fetch_scholarpedia("black holes")
    assert len(result) <= 500
