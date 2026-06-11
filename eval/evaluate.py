"""
Evaluation utilities for SportsMind narrative reports.

Provides lightweight checks for required narrative sections, stat accuracy,
hallucinated player mentions (NER via spaCy), and whether final scores are
mentioned. Exposes evaluate() to produce a structured evaluation summary.

Public API:
- check_sections(report: str) -> tuple[bool, list[str]]
- check_stat_accuracy(report: str, box_score: list[dict]) -> tuple[bool, list[str]]
- check_hallucinated_players(report: str, box_score: list[dict]) -> tuple[bool, list[str]]
- check_score_mentioned(report: str, home_score: int, away_score: int) -> bool
- evaluate(state: SportsMindState) -> dict
- print_eval(result: dict) -> None

Returns (evaluate):
- dict with keys:
  - "game_id" (str)
  - "headline" (str)
  - "confidence_score" (float|int)
  - "sections_ok" (bool)
  - "missing_sections" (list[str])
  - "stats_ok" (bool)
  - "stat_issues" (list[str])
  - "hallucination_ok" (bool)
  - "hallucinated_names" (list[str])
  - "score_mentioned" (bool)
  - "eval_passed" (bool)
  - "eval_score" (str)  # "passed/total"

Raises / Errors:
- spaCy model loading occurs at import and may raise if the model is absent.
- Functions return boolean + issue lists; callers should handle unexpected input shapes.
- For deterministic tests, mock the spaCy pipeline or set up the small model.

Notes:
- REQUIRED_SECTIONS defines the minimal structural elements the narrative should include.
- check_hallucinated_players uses simple substring matching against box_score names;
  consider enhancing fuzzy matching if needed.

Example:
>>> result = evaluate(state)
>>> print(result["eval_score"], result["eval_passed"])
"""

from coordinator.state import SportsMindState
import spacy
nlp = spacy.load("en_core_web_sm")

REQUIRED_SECTIONS = [
    "headline", "summary", "performers", "historical", "turning", "takeaway"
]

def check_sections(report: str) -> tuple[bool, list[str]]:
    report_lower = report.lower()
    keywords = {
        "headline":   ["headline", "**headline"],
        "summary":    ["summary", "game summary"],
        "performers": ["performer", "key performer"],
        "historical": ["historical", "rivalry", "history"],
        "turning":    ["turning point", "turning"],
        "takeaway":   ["takeaway", "take away"]
    }
    missing = [
        section for section, kws in keywords.items()
        if not any(kw in report_lower for kw in kws)
    ]
    return len(missing) == 0, missing

def check_stat_accuracy(report: str, box_score: list[dict]) -> tuple[bool, list[str]]:
    issues = []
    for player in box_score:
        pts = player.get("points")
        if not pts or int(pts) < 20:
            continue
        name = f"{player['firstName']} {player['familyName']}"
        pts_int = int(pts)
        if name in report and str(pts_int) not in report:
            issues.append(f"{name} scored {pts_int} but that number not found in report")
    return len(issues) == 0, issues

def check_hallucinated_players(report: str, box_score: list[dict]) -> tuple[bool, list[str]]:
    known = {
        f"{p['firstName']} {p['familyName']}".lower()
        for p in box_score
    }
    
    # use NER to extract person names
    doc = nlp(report)
    flagged = []
    seen = set()
    
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = ent.text.strip().lower()
        if name in seen:
            continue
        seen.add(name)
        # check if any known player name is a close match
        if not any(
            name in known_name or known_name in name
            for known_name in known
        ):
            flagged.append(ent.text.strip())
    
    return len(flagged) == 0, flagged[:5]

def check_score_mentioned(report: str, home_score: int, away_score: int) -> bool:
    return str(home_score) in report and str(away_score) in report

def evaluate(state: SportsMindState) -> dict:
    report = state["narrative"].report
    box_score = state["stats"].box_score
    home_score = state["stats"].home_score
    away_score = state["stats"].away_score

    sections_ok, missing_sections = check_sections(report)
    stats_ok, stat_issues = check_stat_accuracy(report, box_score)
    hallucination_ok, hallucinated = check_hallucinated_players(report, box_score)
    score_ok = check_score_mentioned(report, home_score, away_score)

    passed = sum([sections_ok, stats_ok, hallucination_ok, score_ok])
    total = 4

    return {
        "game_id":            state["input"].game_id,
        "headline":           state["narrative"].headline,
        "confidence_score":   state["narrative"].confidence_score,
        "sections_ok":        sections_ok,
        "missing_sections":   missing_sections,
        "stats_ok":           stats_ok,
        "stat_issues":        stat_issues,
        "hallucination_ok":   hallucination_ok,
        "hallucinated_names": hallucinated,
        "score_mentioned":    score_ok,
        "eval_passed":        passed == total,
        "eval_score":         f"{passed}/{total}"
    }

def print_eval(result: dict) -> None:
    print("\n=== EVAL RESULTS ===")
    print(f"Game:            {result['game_id']}")
    print(f"Eval Score:      {result['eval_score']} {'✓' if result['eval_passed'] else '✗'}")
    print(f"Confidence:      {result['confidence_score']}/10")
    print(f"Sections:        {'✓' if result['sections_ok'] else '✗ missing: ' + str(result['missing_sections'])}")
    print(f"Stat Accuracy:   {'✓' if result['stats_ok'] else '✗ ' + str(result['stat_issues'])}")
    print(f"Hallucinations:  {'✓' if result['hallucination_ok'] else '✗ flagged: ' + str(result['hallucinated_names'])}")
    print(f"Score Mentioned: {'✓' if result['score_mentioned'] else '✗'}")