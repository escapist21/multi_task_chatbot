from __future__ import annotations

import os
from typing import List

import requests


class TavilyError(RuntimeError):
    pass


def tavily_search_summarize(query: str, max_results: int = 5) -> str:
    """Search Tavily and return a compact, source-linked summary.

    Requires TAVILY_API_KEY in environment.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise TavilyError("TAVILY_API_KEY is not set")
    max_results = max(1, min(int(max_results or 5), 10))

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=20,
        )
    except Exception as e:
        raise TavilyError(f"Request failed: {e}")

    if resp.status_code != 200:
        raise TavilyError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    answer = data.get("answer")
    results: List[dict] = data.get("results") or []

    lines: List[str] = []
    if answer:
        lines.append(answer.strip())
    if results:
        lines.append("")
        lines.append("Sources:")
        for r in results[:max_results]:
            title = r.get("title") or r.get("url") or "source"
            url = r.get("url") or ""
            lines.append(f"- {title} — {url}")
    return "\n".join(lines).strip()


