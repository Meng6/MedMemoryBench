# MedMemoryBench

> A Benchmark Framework for Evaluating Agent Memory Methods on Medical Dialogue

**MedMemoryBench** is a benchmark framework for evaluating Agent memory methods, with a focus on memory capability assessment in medical dialogue scenarios. This framework provides unified evaluation interfaces, multiple baseline method implementations, and a flexible configuration management system, while also supporting the import and evaluation of other datasets.

## 📊 Dataset Overview

MedMemoryBench ships a Chinese medical dialogue dataset (with a parallel English version) generated around long-horizon patient personas:

- **20 personas**, each containing a longitudinal background, life events, and trap events
- **~2,020 dialogue sessions** in total (≈101 per persona), simulating multi-session doctor–patient interactions
- **~1,986 evaluation queries** spanning 6 query types:
  - `entity_exact_match` (400) — factual recall
  - `temporal_localization` (400) — time-related reasoning
  - `multiple_choice` (398) — multi-choice QA
  - `inference_generation` (397) — open-ended inference
  - `state_update` (200) — tracking evolving patient states
  - `multi_hop_clinical_deduction` (191) — multi-hop clinical reasoning
- **Size on disk**: ~598 MB (`data/MedMemoryBench/`, Chinese) + ~443 MB (`data/MedMemoryBench_EN/`, English)
- A bundled [LoCoMo](https://github.com/snap-research/locomo) copy (~18 MB) is provided under `data/locomo/` for cross-benchmark comparison.

## 📁 Project Structure

```
MedMemoryBench/
├── main.py                       # Evaluation entry point
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache License 2.0
├── LEGAL.md                      # Comment-language legal notice
├── .env.example                  # Environment variable template
│
├── configs/                      # Configuration files
│   ├── method_config/            # Per-method YAML configs (gpt-5.1 / qwen3 variants)
│   │   ├── long_context_gpt-5.1.yaml
│   │   ├── embedding_rag_gpt-5.1.yaml
│   │   ├── bm25_rag_gpt-5.1.yaml
│   │   ├── graph_rag_gpt-5.1.yaml
│   │   ├── mem0_gpt-5.1.yaml
│   │   ├── memos_gpt-5.1.yaml
│   │   ├── memrl_gpt-5.1.yaml
│   │   ├── mem1_gpt-5.1.yaml
│   │   ├── amem_gpt-5.1.yaml
│   │   ├── hipporag_gpt-5.1.yaml
│   │   ├── lightmem_gpt-5.1.yaml
│   │   ├── letta_gpt-5.1.yaml
│   │   ├── mirix_gpt-5.1.yaml
│   │   ├── remem_gpt-5.1.yaml
│   │   └── zep_gpt-5.1-chat.yaml
│   └── dataset_config/           # Dataset configurations
│       ├── medmemorybench.yaml
│       └── locomo.yaml
│
├── methods/                      # Memory method implementations
│   ├── base.py                   # BaseAgent abstract class
│   ├── long_context.py           # Long-context baseline
│   ├── embedding_rag.py          # Dense embedding RAG
│   ├── bm25_rag.py               # BM25 sparse RAG
│   ├── graph_rag.py              # Graph-based RAG
│   ├── self_rag.py               # Self-RAG
│   ├── mem0_agent.py             # Mem0 adapter
│   ├── memos_agent.py            # MemOS adapter
│   ├── memrl_agent.py            # MemRL adapter
│   ├── amem_agent.py             # A-MEM adapter
│   ├── hipporag_agent.py         # HippoRAG adapter
│   ├── lightmem_agent.py         # LightMem adapter
│   ├── letta_agent.py            # Letta adapter
│   ├── mirix_agent.py            # MIRIX adapter
│   ├── remem_agent.py            # ReMem adapter
│   ├── zep_agent.py              # Zep Cloud adapter
│   └── <vendored upstream repos> # mem0/, memOS/, MemRL/, amem/, HippoRAG/,
│                                 # LightMem/, letta/, MIRIX/, REMem/, MEM1/,
│                                 # cognee/, memorag/  (third-party sources)
│
├── benchmarks/                   # Dataset evaluation implementations
│   ├── base.py                   # BaseDataset abstract class
│   ├── medmemorybench/           # MedMemoryBench dataset
│   │   ├── dataset.py            # Data loading
│   │   ├── evaluator.py          # Evaluation logic
│   │   └── checkpoint.py         # Checkpoint resumption
│   └── locomo/                   # LoCoMo dataset
│       ├── dataset.py
│       └── evaluator.py
│
├── metrics/                      # Evaluation metrics
│   ├── base.py                   # BaseMetric abstract class
│   ├── string_match.py           # String matching metrics
│   ├── llm_judge.py              # LLM-as-a-Judge metrics
│   └── locomo_metrics.py         # LoCoMo-specific metrics
│
├── src/                          # Core orchestration modules
│   ├── config.py                 # Configuration loader
│   ├── agent.py                  # AgentManager
│   ├── evaluator.py              # Evaluation dispatcher
│   └── result.py                 # Result collection and reporting
│
├── utils/                        # Utility modules
│   ├── llm_client.py             # Unified LLM client
│   ├── tokenizer.py              # Tokenizer helpers
│   ├── templates.py              # Prompt templates
│   ├── prompts_qa.py             # QA prompts
│   ├── prompts_judge.py          # Judge prompts
│   ├── prompts_memorize.py       # Memorization prompts
│   ├── langchain_callback.py     # LangChain callback hooks
│   └── logger.py                 # Logger
│
├── docker/                       # Optional service compose files
│   ├── mirix-init.sql
│   └── mirix-services.yml
│
├── scripts/                      # Helper scripts
│   ├── run_eval.sh               # Evaluation entry script
│   └── mirix-services.sh         # MIRIX service launcher
│
├── data/                         # Datasets (Git LFS)
│   ├── MedMemoryBench/           # Chinese version, ~598 MB
│   ├── MedMemoryBench_EN/        # English version, ~443 MB
│   └── locomo/                   # LoCoMo dataset, ~18 MB
│
├── generation/                   # Dataset generation pipeline (separate sub-project)
├── outputs/                      # Evaluation outputs (gitignored)
├── exp_results/                  # Curated experiment reports
├── logs/                         # Runtime logs (gitignored)
└── results/                      # Method-side caches, e.g. Qdrant / mem cubes (gitignored)
```

## 🚀 Quick Start

### 1. Clone the Repository

This repository ships datasets via Git LFS.

```bash
# Install Git LFS (skip if already installed)
brew install git-lfs                  # macOS
sudo apt-get install git-lfs          # Ubuntu/Debian
# Windows: https://git-lfs.github.com/

git lfs install
git clone https://github.com/AQ-MedAI/MedMemoryBench.git
cd MedMemoryBench
```

### 2. Environment Setup

**Using uv (recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

**Using conda:**

```bash
conda create -n medmemorybench python=3.10
conda activate medmemorybench
pip install -r requirements.txt
```

**Method-specific dependencies:** several memory methods vendor upstream packages under `methods/` (e.g. `methods/mem0/`, `methods/memOS/`). If a method has its own `requirements.txt` or `README`, follow those instructions to enable it.

**Embedding models:** method configs reference local embedding models under `models/` (e.g. `models/bge-small-zh-v1.5`). Download them before running:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5').save('models/bge-small-zh-v1.5')"
```

You can also set `MODELS_DIR` to point to a custom models directory.

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the API keys you intend to use:

```env
# BigModel (OpenAI-compatible, primary endpoint used in this project)
BIGMODEL_API_KEY=your_bigmodel_api_key
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# OpenAI (optional, if you use OpenAI directly)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Zep Cloud (optional, only needed for the Zep agent)
ZEP_API_KEY=your_zep_api_key

# Default model selection
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# Optional: isolate Letta local runtime data (defaults to ~/.letta)
LETTA_DIR=.tmp/letta_runtime
```

Notes:
- For Letta with BigModel, set `BIGMODEL_API_KEY` / `BIGMODEL_BASE_URL` first; the framework maps them to OpenAI-compatible settings internally.
- `LETTA_DIR` is recommended to avoid stale local SQLite metadata from previous runs.

### 4. Run Evaluation

**Via shell script:**

```bash
./scripts/run_eval.sh bm25_rag_gpt-5.1 medmemorybench
```

**Via Python:**

```bash
# Standard run
python main.py -m bm25_rag_gpt-5.1 -d medmemorybench

# Dry run (no real LLM/API calls)
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --dry-run

# Resume from checkpoint
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --resume

# List available methods / datasets
python main.py --list-methods
python main.py --list-datasets
```

> 💡 **Adding a new method?** See [methods/README.md](methods/README.md) for how to extend the framework with new memory methods.

## 🧰 Troubleshooting (Letta + BigModel)

### Symptom: `curl` works, but Letta reports `401` / `forbidden`

Common cause: Letta runtime reads a different credential source than the one used in `curl`.

Check in the same shell session used to run evaluation:

```bash
python -c "from src.config import load_env_config; c=load_env_config(); print(bool(c.bigmodel_api_key or c.openai_api_key), c.bigmodel_base_url or c.openai_base_url)"
```

Then verify both Chat and Embedding endpoints with the same key/base:

```bash
python -c "from openai import OpenAI; from src.config import load_env_config; c=load_env_config(); client=OpenAI(api_key=c.bigmodel_api_key or c.openai_api_key, base_url=c.bigmodel_base_url or c.openai_base_url); print('chat', bool(client.chat.completions.create(model='glm-4-plus', messages=[{'role':'user','content':'ping'}], max_tokens=8).choices)); print('emb', len(client.embeddings.create(model='embedding-3', input='ping').data[0].embedding))"
```

If this passes, your key is valid and Letta should authenticate correctly.

### Symptom: Letta fails with SQLite schema / migration errors

Cause: Letta's default local DB (`~/.letta/sqlite.db`) contains old schema state.

Fix:
- Set `LETTA_DIR` to an isolated project-local path, e.g. `.tmp/letta_runtime`.
- Re-run evaluation so Letta creates a fresh local DB under that directory.

### Symptom: intermittent timeout / handshake timeout

This usually indicates transient network/proxy/TLS instability rather than an invalid API key.

Suggested actions:
- Retry the same command in a quiet period (avoid parallel test runs).
- Ensure proxy settings in the terminal are consistent with your successful `curl` environment.

## 🔧 Configuration

### Method Configuration (`configs/method_config/`)

```yaml
# configs/method_config/embedding_rag_gpt-5.1.yaml

method_name: "embedding_rag"
method_type: "rag"                  # baseline / rag / agentic_memory
description: "Embedding RAG Agent - Dense vector retrieval based RAG method"

model:
  provider: "openai"
  name: "gpt-5.1"
  temperature: 0.3
  max_completion_tokens: 100000

agent_params:
  top_k: 5                          # Number of documents to retrieve
  chunk_size: 512                   # Text chunk size
  chunk_overlap: 50                 # Chunk overlap

embedding:
  provider: "local"                 # openai / local / huggingface
  model: "/path/to/local/model"
```

### Dataset Configuration (`configs/dataset_config/`)

```yaml
# configs/dataset_config/medmemorybench.yaml

dataset_name: "medmemorybench"
description: "Medical dialogue memory evaluation dataset"
language: "zh"

data:
  root_dir: "data/MedMemoryBench"
  sessions_pattern: "persona_{id}/eval/generated_dialogues.json"
  queries_pattern: "persona_{id}/eval/generated_queries.json"

evaluation:
  mode: "independent"               # independent / merged
  evaluation_interval: 10           # Evaluate every N sessions

query_types:
  - name: "entity_exact_match"
    metric: "string_contain"
  - name: "temporal_localization"
    metric: "llm_judge"
  # ... more types
```

## 📄 Output

Evaluation results are saved under `outputs/<method>_<model>/`:

```
outputs/
└── bm25_rag_gpt-5.1/
    ├── eval_medmemorybench_20260330_181703.json    # Detailed results (JSON)
    ├── report_medmemorybench_20260330_181703.txt   # Human-readable report
    └── memory_builds_20260330_181703.json          # Memory build logs
```

## 📜 License

- **Code** in this repository is released under the [Apache License 2.0](LICENSE).
- **Datasets** under `data/MedMemoryBench/` and `data/MedMemoryBench_EN/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Vendored third-party method sources under `methods/` retain their original upstream licenses where present.
- See [LEGAL.md](LEGAL.md) for the source-comment language clause.
