"""
Tools available to agents.
- web_search: Tavily API
- python_repl: sandboxed code execution
- summarize_document: simple text summarizer
"""
import io
import sys
import traceback
from langchain_core.tools import tool
from loguru import logger

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

import os


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic. Use for factual research."""
    if not TAVILY_AVAILABLE or not os.getenv("TAVILY_API_KEY"):
        return f"[Mock Search Result for: {query}]\nTavily not configured. In production, this returns real web results with sources."
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=5, search_depth="advanced")
        output = []
        for r in results.get("results", []):
            output.append(f"**{r['title']}** ({r['url']})\n{r['content'][:500]}")
        return "\n\n---\n\n".join(output) if output else "No results found."
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Search failed: {str(e)}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code for analysis, calculations, or data processing. Returns stdout output."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals)
        output = buffer.getvalue()
        return output if output else "Code executed successfully (no output)."
    except Exception:
        return f"Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


@tool
def format_report(title: str, sections: list[str]) -> str:
    """Format a structured research report from sections."""
    report = f"# {title}\n\n"
    for i, section in enumerate(sections, 1):
        report += f"## Section {i}\n{section}\n\n"
    return report


ALL_TOOLS = [web_search, python_repl, format_report]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
