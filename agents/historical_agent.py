"""
Historical data retrieval agent for SportsMind.

Lightweight helpers to build queries, fetch/cache historical data (Tavily),
ingest into the RAG store, and attach a HistoricalOutput to SportsMindState.

Public API:
- cache_path(query) -> Path
- search_and_ingest(query, source_key, home_team, away_team) -> str
- build_queries(home, away, date, game_context) -> dict[str, str]
- historical_agent(state: SportsMindState) -> dict

Returns (historical_agent):
- dict with keys:
  - "historical" (HistoricalOutput)
  - "quality" (object)
  - "errors" (dict)

Raises:
- Exceptions are caught and recorded to state["errors"]["historical"]; callers should
  ensure required state fields are present.

Example:
>>> out = historical_agent(state)  # use TAVILY_MOCK=true for deterministic tests
"""

import os
import json
import hashlib
from pathlib import Path
from tavily import TavilyClient
from tools.rag import ingest, retrieve
from coordinator.state import SportsMindState, HistoricalOutput

CACHE_DIR = Path(".cache/tavily")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MOCK_MODE = os.getenv("TAVILY_MOCK", "false").lower() == "true"

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MOCK_DATA = {
    "rivalry": "Historic rivalry with multiple playoff meetings.",
    "streaks": "Key players on hot streaks this season.",
    "form": "Both teams playing well entering this matchup."
}

def cache_path(query: str) -> Path:
    key = hashlib.md5(query.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"

def search_and_ingest(query: str, source_key: str, home_team: str, away_team: str) -> str:
    if MOCK_MODE:
        return MOCK_DATA.get(source_key, "mock result")

    cp = cache_path(query)
    if cp.exists():
        raw = json.loads(cp.read_text())["result"]
    else:
        result = client.search(query, max_results=3)
        raw = " ".join(r["content"] for r in result["results"])
        cp.write_text(json.dumps({"query": query, "result": raw}))

    ingest(raw, source=query, home_team=home_team, away_team=away_team)
    return retrieve(query, home_team=home_team, away_team=away_team)

def build_queries(home: str, away: str, date: str, game_context: str) -> dict[str, str]:
    season = date[:4]
    prefix = "NBA playoffs" if "Playoff" in game_context else f"{season} NBA season"
    round_hint = game_context if "Playoff" in game_context else ""
    return {
        "rivalry": f"{home} vs {away} NBA rivalry history {round_hint}".strip(),
        "streaks": f"{home} {away} key players {prefix} performance streaks",
        "form":    f"{home} {away} {prefix} recent results form {round_hint}".strip(),
    }

def historical_agent(state: SportsMindState) -> dict:
    try:
        home         = state["input"].home_team
        away         = state["input"].away_team
        date         = state["input"].game_date
        game_context = state["input"].game_context

        queries = build_queries(home, away, date, game_context)
        results = {}
        queries_used = []

        for key, query in queries.items():
            results[key] = search_and_ingest(query, key, home, away)
            queries_used.append(query)

        state["historical"] = HistoricalOutput(
            rivalry_context=results["rivalry"],
            player_streaks=results["streaks"],
            recent_form=results["form"],
            search_queries_used=queries_used
        )
        state["quality"].historical_ok = True
    except Exception as e:
        state["errors"]["historical"] = str(e)
        state["quality"].historical_ok = False
    return {
        "historical": state["historical"],
        "quality":    state["quality"],
        "errors":     state["errors"],
    }