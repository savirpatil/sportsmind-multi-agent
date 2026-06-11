# SportsMind 🏀

A production-grade multi-agent NBA game analysis system that generates broadcast-quality game reports with built-in hallucination detection and LLM-as-judge eval. Deployed as a streaming web app.

## How it works

```
Date picker → Game selector (NBA schedule API)
       ↓
Game Resolver — fuzzy team matching + playoff round detection
       ↓
LangGraph StateGraph (parallel execution + quality gate + retry)
    ├── Stats Agent ─────────────────┐
    │   NBA API v3 box score +       ├── parallel via fan-out, confirmed in LangSmith
    │   play-by-play                 │
    └── Historical Agent ────────────┘
        Tavily search → ChromaDB RAG (all-MiniLM-L6-v2)
        per-matchup collection scoping prevents context bleed
       ↓
    Momentum Agent
        algorithmic turning point detection from play-by-play
        finds largest unanswered scoring run, quarter, clock time
        grounds Turning Point section in verified data, not hallucinated
       ↓
    Narrative Agent
        Groq llama-3.3-70b-versatile
        prompt loaded from prompts/narrative_v1.txt
        SSE streaming to UI token by token
       ↓
    Quality Gate
        Pydantic validation on all agent outputs
        LLM-as-judge confidence scoring (1-10)
        spaCy NER hallucination detection
        section completeness and stat accuracy checks
        retry failed agents up to MAX_RETRIES
       ↓
Eval badge rendered in UI · W&B auto-logged · LangSmith traced
```

## Quick start

Requirements: Python 3.11+

```bash
git clone https://github.com/savirpatil/sportsmind
cd sportsmind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create `.env`:

```
GROQ_API_KEY=
TAVILY_API_KEY=
LANGCHAIN_API_KEY=
LANGSMITH_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=sportsmind
WANDB_API_KEY=
```

Launch:

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`, pick a date, select a game, watch the report stream.

## Agents

**Stats** — NBA API v3 · box score, play-by-play, team scores

**Historical** — Tavily + ChromaDB · rivalry context and recent form via RAG, scoped per matchup to prevent context bleed between games

**Momentum** — algorithmic · scans play-by-play to find the largest unanswered scoring run, outputs team, point total, quarter, and clock time as ground truth for the narrative

**Narrative** — Llama 3.3 70B via Groq · generates structured report from versioned prompt template, streams token by token via SSE

**Coordinator** — LangGraph · fan-out parallel execution, Pydantic state validation, conditional retry logic

## Engineering decisions

**LangGraph StateGraph** for orchestration — explicit typed state, visualizable graph, native parallel fan-out. State keys use Pydantic BaseModel with Annotated reducers to safely merge concurrent agent writes.

**Algorithmic momentum agent** instead of asking the LLM to infer the turning point — the Turning Point section was the most hallucination-prone part of the report. Running a deterministic scan over play-by-play data and injecting the result into the prompt grounds that section in verified data.

**Per-matchup ChromaDB collections** — a single shared collection caused RAG context bleed between games (Spurs/Lakers history polluting Spurs/Thunder queries). Scoping each collection to a matchup hash fixed this.

**FastAPI SSE streaming** — the full pipeline takes 15-30s. Streaming tokens as they generate makes the latency feel invisible and lets the user see the report build in real time.

**Versioned prompt files** — narrative prompt lives in `prompts/narrative_v1.txt`, loaded at runtime. Swap prompt versions without touching application code.

## Evaluation

Every report is automatically scored on 3 verified dimensions and rendered as a badge in the UI:

- **Section completeness** — keyword matching across 6 required sections
- **Stat accuracy** — cross-reference 20+ point performers against box score
- **Score verification** — final score present in report text

LLM-as-judge confidence score (1-10) is computed using ground-truth stats as context and logged to W&B alongside section scores, stat accuracy, retry count, and home/away scores per run.

spaCy NER occasionally flags team nicknames (e.g. "Spurs") as unverified person entities — known `en_core_web_sm` misclassification, excluded from the UI badge.

## Observability

**LangSmith** — full pipeline traced per run, parallel execution visible in node tree with overlapping timestamps for stats and historical agents

**W&B** — eval metrics and confidence scores auto-logged on every UI request

## Project structure

```
sportsmind/
├── agents/
│   ├── stats_agent.py        # NBA API v3 box score + play-by-play
│   ├── historical_agent.py   # Tavily + ChromaDB RAG, per-matchup scoped
│   ├── momentum_agent.py     # algorithmic turning point from play-by-play
│   └── narrative_agent.py    # Groq streaming, versioned prompt, LLM-as-judge eval
├── coordinator/
│   ├── graph.py              # LangGraph StateGraph, parallel execution, retry
│   └── state.py              # Pydantic models + Annotated reducers for all agent I/O
├── tools/
│   ├── game_resolver.py      # natural language + playoff detection to game_id
│   ├── rag.py                # ChromaDB ingest + retrieval, per-matchup namespacing
│   └── wandb_logger.py       # W&B auto-logging on every UI request
├── eval/
│   └── evaluate.py           # section check, stat accuracy, spaCy NER, score check
├── api/
│   ├── main.py               # FastAPI, async parallel agent calls, SSE streaming
│   └── static/index.html     # date picker, game selector, streaming report, eval badge
├── prompts/
│   └── narrative_v1.txt      # versioned prompt template
└── run.py                    # dev/debug CLI runner
```

## Tech stack

LangGraph · Groq (Llama 3.3 70B) · Tavily · ChromaDB · spaCy · FastAPI · W&B · LangSmith · NBA API