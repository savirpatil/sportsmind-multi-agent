"""
Lightweight W&B run logger for SportsMind.

Initializes a Weights & Biases run, evaluates the produced narrative via
evaluate(), and logs structured metrics and the rendered report HTML.

Public API:
- log_run(state: SportsMindState, game_input: dict) -> None

Returns:
- None (side-effect: remote logging to W&B)

Raises / Errors:
- wandb.init / wandb.log may raise on network/auth failures.
- evaluate(state) may raise or return unexpected shapes; callers should ensure
  state contains "narrative", "stats", and "quality" before calling.

Side effects:
- Starts/finishes a W&B run, sends metrics and HTML payloads (network I/O).

Example:
>>> log_run(state, {"game_id": "1234", "home_team": "LAL", "away_team": "GSW", "game_date": "2024-04-10"})
"""

import os
import wandb
from coordinator.state import SportsMindState
from eval.evaluate import evaluate

def log_run(state: SportsMindState, game_input: dict) -> None:
    wandb.init(
        project="sportsmind",
        name=f"{game_input['away_team']}_at_{game_input['home_team']}_{game_input['game_date']}",
        config={
            "game_id":      game_input["game_id"],
            "home_team":    game_input["home_team"],
            "away_team":    game_input["away_team"],
            "game_date":    game_input["game_date"],
            "model":        "llama-3.3-70b-versatile",
            "rag":          True,
            "is_playoff":   game_input.get("is_playoff", False),
            "game_context": game_input.get("game_context", "NBA regular season game"),
        },
        reinit=True
    )

    eval_result = evaluate(state)

    wandb.log({
        "confidence_score":  state["narrative"].confidence_score,
        "eval_score":        int(eval_result["eval_score"].split("/")[0]),
        "eval_passed":       int(eval_result["eval_passed"]),
        "sections_ok":       int(eval_result["sections_ok"]),
        "stats_ok":          int(eval_result["stats_ok"]),
        "hallucination_ok":  int(eval_result["hallucination_ok"]),
        "score_mentioned":   int(eval_result["score_mentioned"]),
        "momentum_ok":       int(state["quality"].momentum_ok),
        "retries":           state["quality"].retries,
        "home_score":        state["stats"].home_score,
        "away_score":        state["stats"].away_score,
        "is_playoff":        int(game_input.get("is_playoff", False)),
    })

    wandb.log({"report": wandb.Html(
        f"<pre style='font-family:sans-serif'>{state['narrative'].report}</pre>"
    )})

    wandb.finish()