"""
Narrative generation agent for SportsMind.

Builds an LLM prompt from game state, generates a narrative report via ChatGroq,
evaluates the report for quality, and attaches a NarrativeOutput to SportsMindState.

Public API:
- get_llm() -> ChatGroq
- build_prompt(state: SportsMindState) -> str
- score_prompt(report: str, stats) -> str
- evaluate_report(report: str, stats, llm) -> tuple[int, list[str]]
- narrative_agent(state: SportsMindState) -> dict

Returns (narrative_agent):
- dict with keys:
  - "narrative" (NarrativeOutput)
  - "quality" (object)
  - "errors" (dict)

Raises / Errors:
- Exceptions are caught and recorded to state["errors"]["narrative"]; callers should
  ensure state includes stats, input, and any required historical/momentum data.

Example:
>>> out = narrative_agent(state)
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from coordinator.state import SportsMindState, NarrativeOutput
import json
from pathlib import Path

PROMPT_PATH = Path("prompts/narrative_v1.txt")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )

def build_prompt(state: SportsMindState) -> str:
    stats    = state["stats"]
    if not stats:
        raise ValueError("Stats data missing, cannot build narrative")

    historical = state["historical"]
    momentum   = state["momentum"]
    inp        = state["input"]

    top_performers = [
        p for p in stats.box_score
        if p.get("points") and int(p["points"]) >= 20
    ]
    top_str = "\n".join(
        f"- {p['firstName']} {p['familyName']} ({p.get('teamCity','')} {p.get('teamName','')}): "
        f"{p['points']}pts {p.get('reboundsTotal',0)}reb {p.get('assists',0)}ast"
        for p in top_performers
    ) or "No standout performers."

    momentum_section = ""
    if momentum and momentum.run_team:
        momentum_section = (
            f"MOMENTUM ANALYSIS (algorithmically derived from play-by-play — YOU MUST USE THIS):\n"
            f"Largest unanswered run: {momentum.run_score} by the {momentum.run_team} "
            f"in the {momentum.run_quarter}{'st' if momentum.run_quarter == 1 else 'nd' if momentum.run_quarter == 2 else 'rd' if momentum.run_quarter == 3 else 'th'} quarter.\n"
            f"Turning point: {momentum.turning_point}\n"
            f"Details: {momentum.run_description}"
        )

    template = PROMPT_PATH.read_text()
    return template.format(
        away_team=inp.away_team,
        home_team=inp.home_team,
        game_date=inp.game_date,
        game_context=getattr(inp, "game_context", "NBA game"),
        home_score=stats.home_score,
        away_score=stats.away_score,
        top_performers=top_str,
        rivalry_context=historical.rivalry_context[:500],
        recent_form=historical.recent_form[:500],
        player_streaks=historical.player_streaks[:500],
        momentum_section=momentum_section,
    )

def score_prompt(report: str, stats) -> str:
    top = [p for p in stats.box_score if p.get("points") and int(p["points"]) >= 20]
    known_players = [
        f"{p['firstName']} {p['familyName']} ({p.get('teamCity','')} {p.get('teamName','')})"
        for p in top
    ]
    return f"""Rate this NBA game report from 1-10.

GROUND TRUTH (trust this over your training data):
- Known players and teams: {', '.join(known_players)}
- These team assignments are current and correct. Do not flag them as errors.

CRITERIA:
1. Stats accuracy — do numbers match the known players listed above?
2. Sections present — check for: headline, summary, performers, historical, turning point, takeaways (flexible formatting)
3. Professional tone
4. No invented stats or players not in the known list

Report:
{report[:1500]}

Respond with JSON only, no markdown:
{{"score": <1-10>, "issues": ["issue1"]}}"""

def evaluate_report(report: str, stats, llm) -> tuple[int, list[str]]:
    try:
        prompt   = score_prompt(report, stats)
        response = llm.invoke([HumanMessage(content=prompt)])
        clean    = response.content.strip().lstrip("```json").rstrip("```").strip()
        result   = json.loads(clean)
        return result.get("score", 5), result.get("issues", [])
    except Exception:
        return 5, ["Could not evaluate report"]

def narrative_agent(state: SportsMindState) -> dict:
    try:
        llm      = get_llm()
        prompt   = build_prompt(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        report   = response.content
        headline = report.split("\n")[0].strip().lstrip("#").strip()

        score, issues = evaluate_report(report, state["stats"], llm)

        state["narrative"] = NarrativeOutput(
            report=report,
            headline=headline,
            confidence_score=score
        )
        state["quality"].narrative_ok = True

        if score < 5:
            raise ValueError(f"Low confidence score {score}: {issues}")

    except Exception as e:
        state["errors"]["narrative"] = str(e)
        state["quality"].narrative_ok = False

    return {
        "narrative": state["narrative"],
        "quality":   state["quality"],
        "errors":    state["errors"],
    }