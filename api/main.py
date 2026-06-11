"""
FastAPI HTTP API for SportsMind.

Provides endpoints to list games by date and run the full analysis pipeline
(synchronous) or a tokenized streaming analysis (SSE) that runs stats,
historical, momentum, and narrative components.

Public API:
- root() -> HTMLResponse
- games_by_date(body: DateQuery) -> dict: {"games": [...]} where each item has game_id, home_team, away_team, matchup, game_date, is_playoff
- analyze(body: GameQuery) -> dict: {
    "game": GameInput,
    "headline": str,
    "report": str,
    "confidence": Optional[float],
    "eval": dict,
    "quality": dict
  }
- analyze_stream(body: GameQuery) -> StreamingResponse (SSE stream of tokens, eval, done events)

Returns (analyze/analyze_stream):
- analyze: final analysis payload (see keys above).
- analyze_stream: server-sent events yielding JSON objects of type "game", "token", "eval", and "done".

Raises / Errors:
- HTTPException(status_code=400) for invalid client input (e.g., bad date or unresolved query).
- 500 on internal failures; analyze endpoints may raise HTTPException when required pipeline output is missing.
- Pipeline functions may write to logs or external services (wandb); callers should treat API responses as authoritative.

Example:
>>> resp = requests.post("http://localhost:8000/analyze", json={"query": "LAL vs GSW 2024-04-10"}).json()
"""

import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from coordinator.graph import run_pipeline
from tools.game_resolver import resolve_game
from eval.evaluate import evaluate
from tools.wandb_logger import log_run
import json
import time
from nba_api.stats.endpoints import leaguegamefinder
from langsmith import traceable

app = FastAPI(title="SportsMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameQuery(BaseModel):
    query: str

class DateQuery(BaseModel):
    date: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("api/static/index.html") as f:
        return f.read()

@app.post("/games/by-date")
async def games_by_date(body: DateQuery):
    try:
        from datetime import datetime
        parsed   = datetime.strptime(body.date, "%Y-%m-%d")
        api_date = parsed.strftime("%m/%d/%Y")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    loop = asyncio.get_event_loop()
    def _fetch():
        time.sleep(0.6)
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=api_date,
            date_to_nullable=api_date,
            league_id_nullable="00"
        )
        return finder.get_data_frames()[0]

    games_df = await loop.run_in_executor(None, _fetch)

    if games_df.empty:
        return {"games": []}

    seen  = set()
    games = []
    for _, row in games_df.iterrows():
        gid = row["GAME_ID"]
        if gid in seen:
            continue
        seen.add(gid)
        matchup  = row["MATCHUP"]
        opp_rows = games_df[(games_df["GAME_ID"] == gid) & (games_df["TEAM_NAME"] != row["TEAM_NAME"])]
        opp_name = opp_rows.iloc[0]["TEAM_NAME"] if not opp_rows.empty else ""
        if "vs." in matchup:
            home, away = row["TEAM_NAME"], opp_name
        else:
            away, home = row["TEAM_NAME"], opp_name
        is_playoff = gid.startswith("004")
        games.append({
            "game_id":    gid,
            "home_team":  home,
            "away_team":  away,
            "matchup":    f"{away} @ {home}",
            "game_date":  body.date,
            "is_playoff": is_playoff,
        })

    return {"games": games}

@app.post("/analyze")
async def analyze(body: GameQuery):
    try:
        game_input = resolve_game(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = run_pipeline(game_input)
    if not result["narrative"]:
        raise HTTPException(status_code=500, detail=str(result["errors"]))

    eval_result = evaluate(result)
    return {
        "game":       game_input,
        "headline":   result["narrative"].headline,
        "report":     result["narrative"].report,
        "confidence": result["narrative"].confidence_score,
        "eval":       eval_result,
        "quality":    result["quality"].model_dump()
    }

@app.post("/analyze/stream")
async def analyze_stream(body: GameQuery):
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    from agents.narrative_agent import build_prompt
    from coordinator.state import SportsMindState, QualityFlags, GameInput, NarrativeOutput
    from agents.stats_agent import stats_agent
    from agents.historical_agent import historical_agent
    from agents.momentum_agent import momentum_agent

    try:
        game_input = resolve_game(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state: SportsMindState = {
        "input":      GameInput(**game_input),
        "stats":      None,
        "momentum":   None,
        "historical": None,
        "narrative":  None,
        "quality":    QualityFlags(),
        "errors":     {}
    }

    # run stats and historical concurrently in thread pool
    loop = asyncio.get_event_loop()
    stats_result, historical_result = await asyncio.gather(
        loop.run_in_executor(None, stats_agent,      state),
        loop.run_in_executor(None, historical_agent, state),
    )
    state = {**state, **stats_result, **historical_result}
    state = {**state, **momentum_agent(state)}

    async def token_generator():
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
            streaming=True
        )
        prompt = build_prompt(state)
        yield f"data: {json.dumps({'type': 'game', 'data': game_input})}\n\n"

        full_report = ""
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_report += token
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        try:
            headline = full_report.split("\n")[0].strip().lstrip("#").strip()
            state["narrative"] = NarrativeOutput(
                report=full_report,
                headline=headline,
                confidence_score=None
            )
            state["quality"].narrative_ok = True
            eval_result = evaluate(state)
            try:
                log_run(state, game_input)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'eval', 'data': eval_result})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'type': 'eval', 'data': None})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")