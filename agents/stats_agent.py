"""
Stats collection agent for SportsMind.

Fetches box score and play-by-play via nba_api, parses team scores, and
attaches a StatsOutput to SportsMindState.

Public API:
- fetch_box_score(game_id: str) -> list[dict]
- fetch_play_by_play(game_id: str) -> list[dict]
- parse_scores(box_score: list[dict], home_team: str, away_team: str) -> tuple[int, int]
- stats_agent(state: SportsMindState) -> dict

Returns (stats_agent):
- dict with keys:
  - "stats" (StatsOutput)
  - "quality" (object)
  - "errors" (dict)

Raises / Errors:
- Exceptions are caught and recorded to state["errors"]["stats"]; callers should
  ensure state["input"].game_id and state["input"].home_team / away_team exist.

Example:
>>> out = stats_agent(state)
"""

from nba_api.stats.endpoints import boxscoretraditionalv3, playbyplayv3
from nba_api.stats.static import teams
import time
from coordinator.state import SportsMindState, StatsOutput

def fetch_box_score(game_id: str) -> list[dict]:
    time.sleep(0.6)
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    return box.get_data_frames()[0].to_dict(orient="records")

def fetch_play_by_play(game_id: str) -> list[dict]:
    time.sleep(0.6)
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    return pbp.get_data_frames()[0].to_dict(orient="records")

def parse_scores(box_score: list[dict], home_team: str, away_team: str) -> tuple[int, int]:
    team_pts = {}
    for row in box_score:
        city = str(row.get("teamCity", "")).lower()
        name = str(row.get("teamName", "")).lower()
        pts = int(row.get("points", 0) or 0)
        for team in [home_team, away_team]:
            words = [w.lower() for w in team.split()]
            if any(w in city or w in name for w in words):
                team_pts[team] = team_pts.get(team, 0) + pts
    return team_pts.get(home_team, 0), team_pts.get(away_team, 0)

def stats_agent(state: SportsMindState) -> dict:
    try:
        game_id = state["input"].game_id
        box_score = fetch_box_score(game_id)
        play_by_play = fetch_play_by_play(game_id)
        home_score, away_score = parse_scores(
            box_score,
            state["input"].home_team,
            state["input"].away_team
        )
        state["stats"] = StatsOutput(
            box_score=box_score,
            play_by_play=play_by_play,
            home_score=home_score,
            away_score=away_score
        )
        state["quality"].stats_ok = True
    except Exception as e:
        state["errors"]["stats"] = str(e)
        state["quality"].stats_ok = False
    return {
        "stats":   state["stats"],
        "quality": state["quality"],
        "errors":  state["errors"],
    }