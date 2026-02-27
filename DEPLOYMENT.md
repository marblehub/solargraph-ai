# Deployment Guide
## SolarGraph AI — Local · GitHub · HuggingFace Spaces

---

## 1. Final File Structure

```
kg_project/
├── ontology.ttl              # OWL ontology (724 lines, Turtle/RDF)
├── build_graph.py            # parse TTL → graph.pkl
├── query_engine.py           # 15+ SPARQL query methods
├── llm_agent.py              # fast single-shot agent (LRU + file cache)
├── react_agent.py            # ReAct multi-step tool-use agent
├── provenance.py             # triple-level answer provenance
├── ingest_literature.py      # OpenAlex → LLM → RDF pipeline
├── app.py                    # Flask web app (all routes)
├── hf_app.py                 # Gradio app for HuggingFace Spaces
├── visualize.py              # vis.js self-contained graph HTML generator
├── requirements.txt          # pip dependencies
├── .env.example              # environment variable template
├── .gitignore                # excludes graph.pkl, cache.json, .env
├── README.md                 # paper-style documentation
├── DEPLOYMENT.md             # this file
├── LICENSE                   # MIT
├── templates/
│   ├── base.html             # Jinja2 base layout
│   └── home.html             # ReAct toggle + provenance panels
└── static/
    ├── css/styles.css        # ReAct/provenance UI styles added
    └── js/main.js            # textarea auto-resize utility

# Generated at runtime (NOT committed to git):
├── graph.pkl                 # pickled RDFLib graph
├── cache.json                # fast agent answer cache
├── react_cache.json          # ReAct agent answer cache
├── ingested_papers.json      # audit log of ingested papers
└── app.log                   # runtime log
```

---

## 2. Local Setup

### 2a. Ensure all files exist in your project folder
- `app.py`
- `react_agent.py`
- `provenance.py`
- `ingest_literature.py`
- `hf_app.py`
- `requirements.txt`
- `templates/home.html`
- `static/css/styles.css`
- `README.md`
- `.gitignore`
- `LICENSE`
- `DEPLOYMENT.md`

### 2b. Install dependencies

```bash
cd kg_project
source .venv/bin/activate          # or: .venv\Scripts\activate on Windows

pip install -r requirements.txt    # installs requests and gradio (new additions)
```

### 2c. Clear stale cache and rebuild graph

**Always do this after updating any files:**

```bash
rm -f graph.pkl cache.json react_cache.json
python build_graph.py
```

Expected output:
```
✅  PV Solar graph built — XXXX triples saved to graph.pkl
```

### 2d. Test the literature ingestion (optional but recommended)

```bash
# Dry run first (no changes to graph)
python ingest_literature.py --query "perovskite solar cell" --limit 3 --dry-run

# Real ingestion (adds triples to graph, regenerates graph.html)
python ingest_literature.py --query "perovskite solar cell" --limit 5
python ingest_literature.py --query "CIGS thin film" --limit 3
python ingest_literature.py --query "silicon heterojunction" --limit 3
```

### 2e. Start the app (locally, test server)

```bash
python app.py
```

Open http://127.0.0.1:5000

### 2f. Test the new features

**ReAct Agent (UI):** Toggle "ReAct Agent" switch on the home page, ask a question.

**ReAct Agent (API):**
```bash
curl -X POST http://127.0.0.1:5000/ask/react \
  -H "Content-Type: application/json" \
  -d '{"query": "What defects affect perovskite cells and why?"}'
```

**Literature ingestion (API):**
```bash
curl -X POST http://127.0.0.1:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "CIGS solar cell", "limit": 3, "dry_run": true}'
```

---

## 3. GitHub Setup

### 3a. Initialise repository

```bash
cd kg_project
git init
git add .
git commit -m "feat: PV Solar KG with ReAct agent, provenance, and literature ingestion"
```

### 3b. Create GitHub repo and push

```bash
# Create repo at github.com/new (name: solargraph-ai, public, no README)
git remote add origin https://github.com/YOUR_USERNAME/solargraph-ai.git
git branch -M main
git push -u origin main
```

### 3c. Add a GitHub Actions workflow (optional but impressive)

Create `.github/workflows/test.yml`:

```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: python build_graph.py
      - run: python -c "from query_engine import QueryEngine; from build_graph import load_graph; qe = QueryEngine(load_graph()); s = qe.get_graph_summary(); assert s['total_triples'] > 100, f'Too few triples: {s}'; print('✅ KG OK:', s)"
```

### 3d. Future updates workflow

```bash
# Make changes to any file, then:
rm -f graph.pkl cache.json react_cache.json   # clear stale caches
git add .
git commit -m "feat: describe your change"
git push origin main
```

---

## 4. HuggingFace Spaces Deployment

### 4a. Create a new Space

1. Go to https://huggingface.co/new-space
2. Set:
   - **Owner:** your HF username
   - **Space name:** `solargraph-ai`
   - **SDK:** Gradio
   - **Visibility:** Public
3. Click "Create Space"

### 4b. Configure HF to use hf_app.py

HuggingFace auto-detects `app.py` by default. Since our `app.py` is Flask (not Gradio),
we need to tell HF to use `hf_app.py` instead.

Create a file called `README.md` at the **root of the Space repo** (not your project README)
with this header block:

```yaml
---
title: SolarGraph AI
emoji: ☀️
colorFrom: orange
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: hf_app.py
pinned: false
license: mit
short_description: LLM Agent & Knowledge Graph for PV Materials Science
---
```

**Important:** This YAML front-matter must be at the very top of the file, before any other content.

### 4c. Push to HuggingFace

```bash
# Option A: Push from your existing git repo (recommended)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/solargraph-ai
git push hf main

# Option B: Clone HF Space repo and copy files
git clone https://huggingface.co/spaces/YOUR_USERNAME/solargraph-ai hf_space
cp -r kg_project/. hf_space/
cd hf_space
git add . && git commit -m "Initial deploy" && git push
```

### 4d. Add API key as a Secret

1. Go to your Space page → **Settings** tab → **Repository secrets**
2. Click **New Secret**:
   - Name: `GROQ_API_KEY`
   - Value: your key from console.groq.com
3. Click **Save**

HuggingFace injects secrets as environment variables automatically.

### 4e. Wait for build

The Space will build automatically (~2-3 minutes). Watch the build logs in the **Logs** tab.

If it fails, common fixes:
- Make sure `requirements.txt` includes `gradio>=4.0`
- Make sure `hf_app.py` is in the root of the repo
- Check the YAML front-matter in the Space's README.md

### 4f. Future HF updates

```bash
git add .
git commit -m "update: describe change"
git push hf main    # triggers automatic rebuild on HF
```

---

## 5. Key API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page with stats + query UI |
| `POST` | `/ask` | Fast single-shot answer (cached) |
| `POST` | `/ask/react` | ReAct multi-step agent + provenance |
| `GET` | `/graph` | vis.js interactive knowledge graph |
| `GET` | `/api/stats` | Triple + entity counts |
| `GET` | `/api/entities?type=Absorber` | Entity list by OWL class |
| `GET` | `/api/absorbers` | Absorbers with bandgap data |
| `GET` | `/api/architectures` | Cell architectures by efficiency |
| `GET` | `/api/relationships` | All KG triples |
| `GET` | `/api/search?q=perovskite` | Keyword search |
| `GET` | `/api/cache/stats` | Cache hit/miss stats |
| `POST` | `/api/cache/clear` | Clear all caches |
| `POST` | `/api/ingest` | OpenAlex ingestion trigger |

---

## 6. Literature Ingestion Usage

```bash
# Command line
python ingest_literature.py --help

python ingest_literature.py \
  --query "perovskite solar cell stability" \
  --limit 10

python ingest_literature.py \
  --query "CIGS thin film efficiency" \
  --limit 5 \
  --dry-run                          # preview without writing

# REST API (app must be running)
curl -X POST http://127.0.0.1:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "query": "silicon heterojunction solar cell",
    "limit": 5,
    "dry_run": false
  }'
```

The ingestion pipeline:
1. Hits OpenAlex free API (no key required)
2. Reconstructs abstracts from OpenAlex inverted-index format
3. Sends each paper's title + abstract to Groq LLM for structured entity extraction
4. Converts extracted entities into typed RDF triples
5. Adds `lit:mentionedIn` provenance links connecting entities to paper nodes
6. Re-pickles the graph and regenerates `graph.html`
7. Logs every ingested paper to `ingested_papers.json` (skips duplicates on re-run)

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `0 nodes, 0 edges` in graph | Delete `graph.pkl` and restart — old pickle from previous ontology |
| `ModuleNotFoundError: rdflib` | Run `pip install -r requirements.txt` inside venv |
| `GROQ_API_KEY not set` | Add key to `.env` file or HF Secrets |
| Graph HTML 404 lib errors | You have old PyVis-generated HTML — delete `templates/graph.html` and restart |
| `requests` not found | `pip install requests` or reinstall from `requirements.txt` |
| HF Space fails to build | Check `requirements.txt` has `gradio>=4.0`; check YAML front-matter in Space README |
| ReAct agent takes too long | Normal — it runs up to 6 LLM+tool iterations. Results are cached after first run. |
