# Multilingual RAG System

Production-grade Retrieval-Augmented Generation system with dual embedding pipelines (OpenAI + Cohere)

---

## Architecture

```
rag_system/
├── config/          # Pydantic settings, env management
├── ingestion/       # PDF and CSV ingestors
├── chunking/        # Semantic chunker (sentence-transformers)
├── embeddings/      # OpenAI + Cohere embedding pipelines
├── retrieval/       # Retrieval pipeline + model comparison
├── rag/             # Response generator with grounded citations
├── evaluation/      # Recall@K, Precision@K, MRR, NDCG@K, latency
├── api/             # FastAPI routes + Pydantic schemas
├── data/            # Mock datasets + generator script
├── tests/           # Benchmark queries
├── main.py          # Entry point + html frontend integrated 
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required keys in `.env`:**



### 3. Generate mock data

```bash
python data/generate_mock_data.py
```

Creates:
- `data/sales_data.csv` — 120-row English sales dataset
- `data/ventas_datos.csv` — 80-row Spanish financial dataset
- `data/corporate_report_en.pdf` — English corporate report (7 sections)
- `data/informe_corporativo_es.pdf` — Spanish corporate report (7 sections)

### 4. Run

```bash
python main.py
```

Open: `http://localhost:8000`
