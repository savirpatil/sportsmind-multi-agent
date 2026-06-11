"""
Convenience script to run the SportsMind pipeline against example games.

Loads environment variables, runs run_pipeline for a small set of sample
games, prints pipeline quality/errors/headline, and runs evaluate() when a
narrative is produced.

Public API:
- run_pipeline(game_input: dict) -> SportsMindState  # imported from coordinator.graph
- evaluate(state: SportsMindState) -> dict
- print_eval(result: dict) -> None

Returns:
- None (side-effects: prints to stdout and may call external services)

Raises / Errors:
- Exceptions from run_pipeline are caught by the pipeline and surfaced in
  the returned state's "errors" field; the script itself may raise on import
  failures (missing env, deps). Intended as a local/dev helper — not a stable API.

Example:
>>> python run.py
"""

from dotenv import load_dotenv
load_dotenv()

from coordinator.graph import run_pipeline
from eval.evaluate import evaluate, print_eval

GAMES = [
    {
        "game_id":      "0022500523",
        "home_team":    "San Antonio Spurs",
        "away_team":    "Los Angeles Lakers",
        "game_date":    "2026-01-07",
        "image_url":    None,
        "is_playoff":   False,
        "game_context": "NBA regular season game"
    },
    {
        "game_id":      "0042500317",
        "home_team":    "Oklahoma City Thunder",
        "away_team":    "San Antonio Spurs",
        "game_date":    "2026-05-30",
        "image_url":    None,
        "is_playoff":   True,
        "game_context": "NBA Playoff game - Conference Finals Game 7"
    },
]

for game in GAMES:
    print(f"\n{'='*60}")
    print(f"Running: {game['away_team']} @ {game['home_team']} ({game['game_date']})")
    print('='*60)

    result = run_pipeline(game)

    print(f"Quality: {result['quality']}")
    print(f"Errors:  {result['errors']}")
    print(f"Headline: {result['narrative'].headline if result['narrative'] else 'FAILED'}")

    if result["narrative"]:
        eval_result = evaluate(result)
        print_eval(eval_result)