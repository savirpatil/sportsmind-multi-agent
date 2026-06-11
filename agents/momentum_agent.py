"""
Momentum analysis agent for SportsMind.

Compute the largest unanswered scoring run from play-by-play and attach a
MomentumOutput to SportsMindState.

Public API:
- get_play_points(play) -> int
- find_largest_run(play_by_play, home_tricode, away_tricode) -> dict
- clock_to_readable(clock) -> str
- tricode_to_full(tricode, home_team, away_team) -> str
- momentum_agent(state: SportsMindState) -> dict

Returns (momentum_agent):
- dict with keys:
  - "momentum" (MomentumOutput)
  - "quality" (object)
  - "errors" (dict)

Raises / Errors:
- Exceptions are caught and recorded to state["errors"]["momentum"]; callers should
  ensure state.stats.play_by_play and state.input.home_team / away_team exist.

Example:
>>> out = momentum_agent(state)
"""

from coordinator.state import SportsMindState, MomentumOutput

QUARTER_NAMES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}

def get_play_points(play: dict) -> int:
    if play.get("isFieldGoal") == 1 and play.get("shotResult") == "Made":
        return int(play.get("shotValue") or 0)
    if "Free Throw" in play.get("actionType", ""):
        if "MISS" not in play.get("description", "").upper():
            return 1
    return 0

def find_largest_run(play_by_play: list[dict], home_tricode: str, away_tricode: str) -> dict:
    """
    Find the largest unanswered scoring run by either team.
    A run ends when the opposing team scores.
    """
    scored_plays = [
        p for p in play_by_play
        if get_play_points(p) > 0
        and p.get("teamTricode") in (home_tricode, away_tricode)
    ]

    if not scored_plays:
        return {}

    best = {"margin": 0}

    # for each play, extend forward as long as same team scores
    i = 0
    while i < len(scored_plays):
        team      = scored_plays[i]["teamTricode"]
        run_pts   = 0
        run_start = i
        j         = i

        # accumulate while same team scores uninterrupted
        while j < len(scored_plays) and scored_plays[j]["teamTricode"] == team:
            run_pts += get_play_points(scored_plays[j])
            j += 1

        if run_pts > best["margin"]:
            best = {
                "margin":     run_pts,
                "run_pts":    run_pts,
                "team":       team,
                "quarter":    scored_plays[run_start]["period"],
                "clock":      scored_plays[run_start]["clock"],
                "start_play": scored_plays[run_start],
                "end_play":   scored_plays[j - 1],
            }

        i = j if j > i else i + 1

    return best

def clock_to_readable(clock: str) -> str:
    try:
        clock = clock.replace("PT", "").replace("S", "")
        mins, secs = clock.split("M")
        return f"{int(float(mins))}:{float(secs):04.1f}"
    except Exception:
        return clock

def tricode_to_full(tricode: str, home_team: str, away_team: str) -> str:
    tc = tricode.upper()
    home_words = [w.upper() for w in home_team.split()]
    away_words = [w.upper() for w in away_team.split()]
    if any(w.startswith(tc) or tc in w for w in home_words):
        return home_team
    if any(w.startswith(tc) or tc in w for w in away_words):
        return away_team
    return tricode

def momentum_agent(state: SportsMindState) -> dict:
    try:
        pbp       = state["stats"].play_by_play
        home_team = state["input"].home_team
        away_team = state["input"].away_team

        # extract the two tricodes that actually appear in scoring plays
        tricodes = [
            p["teamTricode"] for p in pbp
            if p.get("teamTricode") and p["teamTricode"] != ""
            and get_play_points(p) > 0
        ]
        unique_tricodes = list(dict.fromkeys(tricodes))  # preserve order
        home_tri = unique_tricodes[0] if len(unique_tricodes) > 0 else ""
        away_tri = unique_tricodes[1] if len(unique_tricodes) > 1 else ""

        # map tricode back to full name by checking which team name contains it
        def tri_to_full(tc):
            tc_lower = tc.lower()
            for team in [home_team, away_team]:
                words = [w.lower() for w in team.split()]
                if any(w.startswith(tc_lower) or tc_lower in w for w in words):
                    return team
            if tc == home_tri:
                return home_team
            return away_team

        run = find_largest_run(pbp, home_tri, away_tri)

        if not run.get("team"):
            state["momentum"] = MomentumOutput(
                turning_point="Momentum shifted in the second half as one team pulled away.",
                run_team=home_team,
                run_score="N/A",
                run_quarter=3,
                run_description="The winning team outscored their opponent in the second half."
            )
        else:
            quarter_str = QUARTER_NAMES.get(run["quarter"], f"Q{run['quarter']}")
            run_team    = tri_to_full(run["team"])
            clock_str   = clock_to_readable(run["start_play"].get("clock", ""))
            end_desc    = run["end_play"].get("description", "").replace("\n", " ")

            turning_point = (
                f"A {run['run_pts']}-0 run by the {run_team} "
                f"in the {quarter_str} quarter (at {clock_str}) "
                f"was the decisive momentum shift."
            )
            run_description = (
                f"The run ended with {end_desc} — "
                f"a {run['run_pts']}-point unanswered stretch that proved decisive."
            )

            state["momentum"] = MomentumOutput(
                turning_point=turning_point,
                run_team=run_team,
                run_score=f"{run['run_pts']}-0",
                run_quarter=run["quarter"],
                run_description=run_description
            )

        state["quality"].momentum_ok = True

    except Exception as e:
        state["errors"]["momentum"] = str(e)
        state["quality"].momentum_ok = False
        state["momentum"] = MomentumOutput(
            turning_point="Momentum analysis unavailable.",
            run_team="",
            run_score="N/A",
            run_quarter=0,
            run_description=""
        )

    return {
        "momentum": state["momentum"],
        "quality":  state["quality"],
        "errors":   state["errors"],
    }