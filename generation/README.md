# MedMemoryBench &mdash; Data Generation Pipeline

Automated pipeline for synthesizing the **MedMemoryBench** dataset: persona-grounded medical dialogue histories with structured evaluation queries and controllable noise augmentation.

---

## Directory Structure

```
generation/
├── run.sh                        # Main 6-stage pipeline entry point
├── pyproject.toml                # Project metadata & dependencies
├── .env / .env.example           # LLM & database configuration
│
├── app/                          # Core application layer
│   ├── config.py                 #   Environment settings (LLM, DB, server)
│   ├── database.py               #   Async SQLite via SQLAlchemy
│   ├── models/                   #   ORM models (persona, event, dialogue, task)
│   ├── schemas/                  #   Pydantic schemas (persona, event, dialogue, query)
│   ├── services/                 #   Business logic (LLM calls, token tracking, generation)
│   ├── prompts/                  #   All LLM prompt templates
│   └── routers/                  #   FastAPI routes (optional REST API)
│
├── pipeline/                     # Batch generation pipeline (CLI)
│   ├── cli.py                    #   CLI entry & argument parsing
│   ├── config.py                 #   Pipeline-level configs & presets
│   ├── generator.py              #   Orchestrates persona / event / dialogue generation
│   ├── query_generator.py        #   Evaluation query generation (6 query types)
│   ├── exporters.py              #   JSON export utilities
│   └── examples.py               #   Few-shot examples for prompts
│
├── augmentation/                 # Post-generation augmentation
│   ├── noise/                    #   Type-1 noise: health-knowledge chit-chat sessions
│   ├── noise_family/             #   Type-2 noise: family-member consultation sessions
│   ├── run_noise.sh              #   Noise generation & injection script
│   ├── check/                    #   Query difficulty validation (memory-free model test)
│   └── check.sh                  #   Difficulty check entry script
│
├── data/                         # Seed data & intermediate outputs
│   ├── user_personas.json        #   Base persona definitions
│   └── user_report.json          #   Clinical report templates
│
└── dataset/                      # Final per-persona dataset output
    └── persona_{id}/
        ├── background/           #   Persona profile, events, trap events, noise sessions, DB
        └── eval/                 #   Dialogues, noised dialogues, evaluation queries
```

---

## Pipeline Overview

The generation process is a **6-stage sequential pipeline** orchestrated by `run.sh`, followed by optional augmentation steps.

```
 Seed Personas ──► Enriched Personas ──► Trap Events ──► Regular Events
                                                              │
                                          Eval Queries ◄── Dialogues
                                               │
                                     [Noise Augmentation] (optional)
                                               │
                                     [Difficulty Check]   (optional)
```

| Stage | Command | Description |
|:-----:|---------|-------------|
| 1 | `import-personas` | Load base personas from JSON into SQLite |
| 2 | `generate-personas` | Enrich base personas into detailed medical profiles via LLM |
| 3 | `generate-trap-events` | Create 6 types of confusable trap events per persona |
| 4 | `generate-regular-events` | Generate phased clinical event timelines around trap events |
| 5 | `generate-dialogues` | Simulate multi-turn doctor-patient dialogue sessions |
| 6 | `generate-queries` | Produce evaluation queries (EEM, TLA, SUA, MQ, IG, MCD) from dialogues |

---

## Usage

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then fill in your API key
```

### Running the Main Pipeline

```bash
# Full pipeline for a single persona (edit TARGET_PERSONA_ID in run.sh)
bash run.sh
```

#### `run.sh` Configuration Variables

| Section | Variable | Default | Description |
|---------|----------|---------|-------------|
| **Target** | `TARGET_PERSONA_ID` | `15` | Specific persona to generate (leave empty to batch-generate) |
| **Persona** | `PERSONA_COUNT` | `1` | Number of personas when not targeting a specific ID |
| | `PERSONA_CONCURRENCY` | `3` | Parallel generation workers |
| **Event** | `EVENT_EVENTS_PER_PHASE` | `19` | Events generated per clinical phase |
| | `EVENT_MAX_TOTAL` | `101` | Maximum total events per persona |
| **Dialogue** | `DIALOGUE_SESSIONS` | `101` | Dialogue sessions per persona |
| | `DIALOGUE_TURNS` | `8` | Max turns per dialogue session |
| **Query** | `QUERY_NUM_{EEM,TLA,SUA,MQ,IG,MCD}` | varies | Count of each query type per generation window |
| | `QUERY_GENERATE_EVERY` | `10` | Generate queries every N sessions |
| | `QUERY_MCD_GENERATE_EVERY` | `20` | Generate MCD queries every N sessions |
| **LLM** | `*_TEMPERATURE` | `1.0` | Sampling temperature for each stage |
| | `*_MAX_TOKENS` | varies | Max generation tokens for each stage |

### Noise Augmentation (Optional)

```bash
# Generate and inject both noise types for a specific persona
PERSONA_ID=1 bash augmentation/run_noise.sh all
```

| Command | Description |
|---------|-------------|
| `generate-health` | Generate Type-1 noise (unrelated health chit-chat) |
| `generate-family` | Generate Type-2 noise (family member consultations) |
| `inject-health` | Interleave Type-1 noise into dialogue sessions |
| `inject-family` | Interleave Type-2 noise into dialogue sessions |
| `all` | Run all four steps sequentially |

### Query Difficulty Check (Optional)

```bash
bash augmentation/check.sh --persona 1
```

Validates generated queries by testing whether a **memory-free** model can answer them correctly. High correct rate indicates the questions are too easy and not sufficiently memory-dependent.


## Environment Variables

See [`.env.example`](.env.example) for the full template. Key variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | LLM provider API key |
| `OPENAI_BASE_URL` | API endpoint (supports any LiteLLM-compatible provider) |
| `LLM_MODEL` | Primary model (default: `gpt-4o`) |
| `DATABASE_URL` | SQLite connection string |
