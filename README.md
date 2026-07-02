# 💬🧠📎 Tracey.
Think: _Clippy_... but for GNW traces. A suite of tools intended to make GNW traces accessible, regardless of technical capability. 

**ℹ️ What this tool does**
Tracey makes GNW agent traces accessible, regardless of technical capability.
- **📊 Explore** Analytics, Trace Explorer, and the Conversation Browser — these read
  **server-side, per-turn aggregations from the Zeno API** (project-zeno).
- **🧪 Sample** for **Human Eval** and **Product Intelligence** — these still pull full
  traces directly from **Langfuse** (fetched once from the sidebar, reused across those tabs).
- **📋 Generate** reports & understand user behaviour.

> Langfuse remains the source of truth for raw traces. The Zeno API just serves the derived
> aggregations (and fetches an individual full trace on demand when you open one in the
> Trace Explorer).

## Run locally

### Prereqs

- Python `>=3.11`
- Recommended: [`uv`](https://github.com/astral-sh/uv)

### 1) Set environment variables

Create a `.env` file in the repo root:

```bash
# Langfuse — source of truth for raw traces; powers Human Eval & Product Intelligence.
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_BASE_URL="..."

# Zeno API — powers the read-path tabs (Analytics, Trace Explorer, Conversation Browser)
# via server-side, per-turn aggregation. The token is a SUPERUSER Resource Watch bearer
# token with read access to the traces APIs.
# NOTE: api.staging.… resolves; staging.api.… does NOT — don't swap them.
ZENO_API_URL="https://api.staging.globalnaturewatch.org"
ZENO_API_TOKEN="..."

# Optional (only needed for Gemini-powered features)
GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
```

### 2) Install dependencies

#### Option A: uv (recommended)

```bash
uv sync
```

#### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### 3) Run the Streamlit app

```bash
uv run streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- **Analytics**, **Trace Explorer**, and **Conversation Browser** fetch on demand from the
  **Zeno API** (their own "🚀 Fetch…" buttons), using the date range / filters in the sidebar.
  Per-turn token/tool/dataset numbers here are computed server-side and are **lower and more
  correct** than the old in-memory aggregation (which summed whole conversations).
- **Human Eval** and **Product Intelligence** fetch the full trace set **once** from the
  sidebar (Langfuse) and reuse it across those tabs.
- Human eval exports are always available via the **Download CSV** button.

## Product Development Mining

The **🧠 Product intelligence** tab contains **Product Development Mining**, split into three sub-tabs:

- **Evidence Mining**: search for traces that support a hypothesis (LLM-scored relevance).
- **Tagging**: LLM-as-judge tagging for prompt topics/flavours and other criteria.
- **Gap Analysis**: generate a markdown report on user jobs, coverage, and gaps.

### LLM settings

In that tab, open **⚙️ LLM Settings** to configure:

- Gemini model
- Optional batching (**Batch traces per Gemini request**, batch size, max chars per trace)

### Editable prompts

Each sub-tab exposes an **📝 Edit system prompt** expander so you can inspect and tweak the prompts used for Gemini.

## Tests
Limited number of tests are provided for charts and user segments. To run them:

```bash
uv run --with pytest pytest tests/test_charts.py -v
```