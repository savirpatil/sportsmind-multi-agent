"""
Pydantic models and state schema for SportsMind pipeline.

Defines input/output models, validation rules, quality flags, and the
SportsMindState TypedDict used to carry data between pipeline nodes.

Public API:
- GameInput: validated input model (game_id, teams, date, etc.)
- StatsOutput: box_score, play_by_play, home_score, away_score (with post-validation)
- MomentumOutput, HistoricalOutput, NarrativeOutput: agent outputs
- QualityFlags: boolean readiness flags and retry counter
- SportsMindState: TypedDict combining the above for pipeline execution
- merge_quality(a, b) -> QualityFlags
- merge_errors(a, b) -> dict
- keep_last(a, b): helper used in state annotations

Returns / Usage:
- These models are instantiated and attached to state; run_pipeline and agents
  consume/produce instances of these types (e.g., state["stats"] = StatsOutput(...)).

Raises / Errors:
- Model validators raise ValueError for invalid data (e.g., bad game_date,
  empty box_score, or incomplete narrative sections). Callers should catch and
  record validation errors into the state's "errors" map.

Example:
>>> gi = GameInput(game_id="0022400001", home_team="LAL", away_team="GSW", game_date="2024-04-10")
>>> state = {"input": gi, "stats": None, "momentum": None, "historical": None, "narrative": None, "quality": QualityFlags(), "errors": {}}
"""

from typing import Optional, Annotated
from pydantic import BaseModel, field_validator, model_validator
from typing import TypedDict

class GameInput(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    image_url: Optional[str] = None
    is_playoff: bool = False
    game_context: str = "NBA regular season game"

    @field_validator("game_date")
    @classmethod
    def validate_date(cls, v):
        import re
        if not re.match(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("game_date must be YYYY-MM-DD")
        return v

    @field_validator("game_id")
    @classmethod
    def validate_game_id(cls, v):
        if not v.isdigit() or len(v) != 10:
            raise ValueError("game_id must be a 10-digit string")
        return v

class StatsOutput(BaseModel):
    box_score: list[dict]
    play_by_play: list[dict]
    home_score: int
    away_score: int

    @model_validator(mode="after")
    def validate_scores(self):
        if self.home_score < 0 or self.away_score < 0:
            raise ValueError("Scores cannot be negative")
        if self.home_score == 0 and self.away_score == 0:
            raise ValueError("Both scores are zero — likely a parse failure")
        if not self.box_score:
            raise ValueError("box_score is empty")
        return self

class MomentumOutput(BaseModel):
    turning_point: str
    run_team: str
    run_score: str
    run_quarter: int
    run_description: str

class HistoricalOutput(BaseModel):
    rivalry_context: str
    player_streaks: str
    recent_form: str
    search_queries_used: list[str]

    @model_validator(mode="after")
    def validate_content(self):
        if not self.rivalry_context or not self.recent_form:
            raise ValueError("Historical context fields are empty")
        return self

class NarrativeOutput(BaseModel):
    report: str
    headline: str
    confidence_score: Optional[int] = None

    @model_validator(mode="after")
    def validate_report(self):
        required_sections = [
            "Key Performers",
            "Historical Context",
            "Turning Point",
            "Takeaways"
        ]
        missing = [s for s in required_sections if s not in self.report]
        if missing:
            raise ValueError(f"Report missing sections: {missing}")
        return self

class QualityFlags(BaseModel):
    stats_ok: bool = False
    momentum_ok: bool = False
    historical_ok: bool = False
    narrative_ok: bool = False
    retries: int = 0

def merge_quality(a: QualityFlags, b: QualityFlags) -> QualityFlags:
    return QualityFlags(
        stats_ok=a.stats_ok or b.stats_ok,
        momentum_ok=a.momentum_ok or b.momentum_ok,
        historical_ok=a.historical_ok or b.historical_ok,
        narrative_ok=a.narrative_ok or b.narrative_ok,
        retries=max(a.retries, b.retries),
    )

def merge_errors(a: dict, b: dict) -> dict:
    return {**a, **b}

def keep_last(a, b):
    return b if b is not None else a

class SportsMindState(TypedDict):
    input:      GameInput
    stats:      Annotated[Optional[StatsOutput],    keep_last]
    momentum:   Annotated[Optional[MomentumOutput], keep_last]
    historical: Annotated[Optional[HistoricalOutput], keep_last]
    narrative:  Annotated[Optional[NarrativeOutput],  keep_last]
    quality:    Annotated[QualityFlags, merge_quality]
    errors:     Annotated[dict[str, str], merge_errors]