"""
Game resolver utilities to parse freeform queries into NBA game metadata.

Provides fuzzy team matching, date parsing, playoff inference, and a resolve_game
helper that queries the nba_api to find the canonical game row.

Public API:
- get_all_teams() -> dict[str, str]
- fuzzy_match_team(query: str, all_teams: dict) -> str
- infer_playoff_round(game_id: str, game_date: str) -> str
- resolve_game(query: str) -> dict

Returns (resolve_game):
- dict with keys:
  - "game_id", "home_team", "away_team", "game_date", "image_url", "is_playoff", "game_context"

Raises / Errors:
- ValueError on unparsable dates or unmatched teams/games.
- nba_api calls may raise or return empty frames; callers should handle these cases.

Notes:
- Date parsing supports several common formats and will default missing year to current year.
- fuzzy_match_team uses SequenceMatcher and a small boost when query words appear in team names.

Example:
>>> resolve_game("Lakers vs Warriors April 10 2024")
{"game_id": "0022400001", "home_team": "Los Angeles Lakers", ...}
"""

import time
from difflib import SequenceMatcher
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams

def get_all_teams() -> dict[str, str]:
    return {
        t["full_name"].lower(): t["full_name"]
        for t in nba_teams.get_teams()
    }

def fuzzy_match_team(query: str, all_teams: dict) -> str:
    query = query.lower().strip()
    best_match, best_score = None, 0
    for key, full_name in all_teams.items():
        score = SequenceMatcher(None, query, key).ratio()
        if any(word in key for word in query.split()):
            score += 0.3
        if score > best_score:
            best_score = score
            best_match = full_name
    return best_match

def infer_playoff_round(game_id: str, game_date: str) -> str:
    """
    Playoff game IDs: 004YYYY0RRG
      R = round (1=first, 2=second, 3=conf finals, 4=finals)
      G = game number
    """
    try:
        round_digit = int(game_id[8])
        game_num    = int(game_id[9])
        rounds = {
            1: "First Round",
            2: "Second Round",
            3: "Conference Finals",
            4: "NBA Finals"
        }
        round_name = rounds.get(round_digit, "Playoff")
        return f"NBA Playoff game - {round_name} Game {game_num}"
    except Exception:
        return "NBA Playoff game"

def resolve_game(query: str) -> dict:
    import re
    from datetime import datetime

    all_teams = get_all_teams()

    date_patterns = [
        r'(\w+ \d{1,2},?\s*\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\w+ \d{1,2})'
    ]
    date_str = None
    for pattern in date_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            break

    date_formats = ["%B %d %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%B %d"]
    parsed_date = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt)
            if parsed_date.year == 1900:
                parsed_date = parsed_date.replace(year=datetime.now().year)
            break
        except Exception:
            continue

    if not parsed_date:
        raise ValueError(f"Could not parse date from: {query}")

    formatted_date = parsed_date.strftime("%Y-%m-%d")
    api_date = parsed_date.strftime("%m/%d/%Y")

    teams_part = re.split(r'\b(?:vs\.?|at|@)\b', query, flags=re.IGNORECASE)[0]
    remaining  = re.split(r'\b(?:vs\.?|at|@)\b', query, flags=re.IGNORECASE)

    team1_query = re.sub(r'\d.*', '', teams_part).strip()
    team2_query = re.sub(r'\d.*', '', remaining[1]).strip() if len(remaining) > 1 else ""

    team1 = fuzzy_match_team(team1_query, all_teams)
    team2 = fuzzy_match_team(team2_query, all_teams)

    if not team1 or not team2:
        raise ValueError(f"Could not resolve teams from: {query}")

    time.sleep(0.6)
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=api_date,
        date_to_nullable=api_date,
        league_id_nullable="00"
    )
    games = finder.get_data_frames()[0]

    if games.empty:
        raise ValueError(f"No games found on {formatted_date}")

    t1_words = [w.lower() for w in team1.split()]
    t2_words = [w.lower() for w in team2.split()]

    matched = games[
        games["TEAM_NAME"].apply(lambda x: any(w in x.lower() for w in t1_words)) |
        games["TEAM_NAME"].apply(lambda x: any(w in x.lower() for w in t2_words))
    ]

    game_ids = matched["GAME_ID"].unique()
    if not len(game_ids):
        raise ValueError(f"Could not match {team1} vs {team2} on {formatted_date}")

    game_id   = game_ids[0]
    game_row  = matched[matched["GAME_ID"] == game_id].iloc[0]
    matchup   = game_row["MATCHUP"]

    if "vs." in matchup:
        home_team = team1 if any(w in matchup.split("vs.")[0].lower() for w in t1_words) else team2
        away_team = team2 if home_team == team1 else team1
    else:
        away_team = team1 if any(w in matchup.split("@")[0].lower() for w in t1_words) else team2
        home_team = team2 if away_team == team1 else team1

    is_playoff   = game_id.startswith("004")
    game_context = infer_playoff_round(game_id, formatted_date) if is_playoff else "NBA regular season game"

    return {
        "game_id":      game_id,
        "home_team":    home_team,
        "away_team":    away_team,
        "game_date":    formatted_date,
        "image_url":    None,
        "is_playoff":   is_playoff,
        "game_context": game_context,
    }