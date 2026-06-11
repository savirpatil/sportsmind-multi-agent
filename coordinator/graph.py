"""
Pipeline graph for SportsMind.

Orchestrates agent nodes (stats, historical, momentum, narrative) using a
StateGraph and exposes run_pipeline to execute the full analysis for a game.

Public API:
- initialize_state(state: SportsMindState) -> dict
- fan_out(state: SportsMindState) -> dict
- run_stats(state: SportsMindState) -> dict
- run_historical(state: SportsMindState) -> dict
- run_momentum(state: SportsMindState) -> dict
- run_narrative(state: SportsMindState) -> dict
- quality_gate(state: SportsMindState) -> dict
- route(state: SportsMindState) -> str
- fail_node(state: SportsMindState) -> dict
- join(state: SportsMindState) -> dict
- build_graph() -> compiled graph
- run_pipeline(game_input: dict) -> SportsMindState

Returns (run_pipeline):
- SportsMindState-like dict with keys:
  - "input" (GameInput)
  - "stats" (StatsOutput | None)
  - "momentum" (MomentumOutput | None)
  - "historical" (HistoricalOutput | None)
  - "narrative" (NarrativeOutput | None)
  - "quality" (QualityFlags)
  - "errors" (dict)

Raises / Errors:
- Node-level exceptions are expected to be recorded into state["errors"] by agents.
- run_pipeline may raise if the underlying graph invocation fails; callers should
  treat the returned state's "errors" and quality flags as the primary signal.

Example:
>>> out = run_pipeline({"game_id": "1234", "home_team": "LAL", "away_team": "GSW"})
"""

from langgraph.graph import StateGraph, END
from langsmith import traceable
from coordinator.state import SportsMindState, QualityFlags, GameInput
from agents.stats_agent import stats_agent
from agents.historical_agent import historical_agent
from agents.momentum_agent import momentum_agent
from agents.narrative_agent import narrative_agent
import time
from tools.wandb_logger import log_run

MAX_RETRIES = 1

def initialize_state(state: SportsMindState) -> dict:
    return {
        "quality":    QualityFlags(),
        "errors":     {},
        "stats":      None,
        "momentum":   None,
        "historical": None,
        "narrative":  None,
    }

def fan_out(state: SportsMindState) -> dict:
    return {}

def run_stats(state: SportsMindState) -> dict:
    return stats_agent(state)

def run_historical(state: SportsMindState) -> dict:
    return historical_agent(state)

def run_momentum(state: SportsMindState) -> dict:
    return momentum_agent(state)

def run_narrative(state: SportsMindState) -> dict:
    if state["quality"].retries > 0:
        time.sleep(15)
    return narrative_agent(state)

def quality_gate(state: SportsMindState) -> dict:
    return {}

def route(state: SportsMindState) -> str:
    q = state["quality"]
    if not q.stats_ok:
        if q.retries < MAX_RETRIES:
            state["quality"].retries += 1
            return "retry_stats"
        return "fail"
    if not q.historical_ok:
        if q.retries < MAX_RETRIES:
            state["quality"].retries += 1
            return "retry_historical"
        return "fail"
    if not q.narrative_ok:
        if q.retries < MAX_RETRIES:
            state["quality"].retries += 1
            return "retry_narrative"
        return "fail"
    return "complete"

def fail_node(state: SportsMindState) -> dict:
    print(f"Pipeline failed. Errors: {state['errors']}")
    return {}

def join(state: SportsMindState) -> dict:
    return {}

def build_graph():
    graph = StateGraph(SportsMindState)

    graph.add_node("initialize",   initialize_state)
    graph.add_node("fan_out",      fan_out)
    graph.add_node("stats",        run_stats)
    graph.add_node("historical",   run_historical)
    graph.add_node("momentum",     run_momentum)
    graph.add_node("join",         join)
    graph.add_node("narrative",    run_narrative)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("fail",         fail_node)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize",  "fan_out")
    graph.add_edge("fan_out",     "stats")
    graph.add_edge("fan_out",     "historical")
    graph.add_edge("stats",       "momentum")
    graph.add_edge("momentum",    "join")
    graph.add_edge("historical",  "join")
    graph.add_edge("join",        "narrative")
    graph.add_edge("narrative",   "quality_gate")
    graph.add_edge("fail",        END)

    graph.add_conditional_edges(
        "quality_gate",
        route,
        {
            "complete":         END,
            "fail":             "fail",
            "retry_stats":      "stats",
            "retry_historical": "historical",
            "retry_narrative":  "narrative",
        }
    )

    return graph.compile()

@traceable(name="sportsmind_pipeline")
def run_pipeline(game_input: dict) -> SportsMindState:
    graph = build_graph()
    initial_state: SportsMindState = {
        "input":      GameInput(**game_input),
        "stats":      None,
        "momentum":   None,
        "historical": None,
        "narrative":  None,
        "quality":    QualityFlags(),
        "errors":     {}
    }
    result = graph.invoke(initial_state)
    if result["narrative"]:
        log_run(result, game_input)
    return result