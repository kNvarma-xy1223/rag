# Multilingual RAG System

Production-grade Retrieval-Augmented Generation system with dual embedding pipelines (OpenAI + Cohere), local Qdrant vector storage, semantic chunking, grounded citations, and evaluation metrics.

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

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | HTML frontend |
| `POST` | `/api/ingest` | Upload PDF/CSV, embed, index |
| `POST` | `/api/query` | RAG query with grounded answer |
| `POST` | `/api/retrieve` | Raw chunk retrieval (no generation) |
| `POST` | `/api/compare` | Side-by-side OpenAI vs Cohere retrieval |
| `POST` | `/api/evaluate` | Single query evaluation metrics |
| `POST` | `/api/benchmark` | Batch benchmark with aggregated metrics |
| `GET` | `/api/collections` | Qdrant collection stats |
| `DELETE` | `/api/collections/{model}` | Clear collection(s) |

### Ingest

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@data/corporate_report_en.pdf" \
  -F "embedding_model=both"
```

### Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Q2 2024 total revenue?",
    "embedding_model": "openai",
    "top_k": 5,
    "filters": {"source_type": "pdf"}
  }'
```

### Compare

```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cuál es el margen EBITDA?", "top_k": 5}'
```

### Benchmark

```bash
curl -X POST http://localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "benchmark_queries": [
      {"query": "Total revenue Q2 2024", "relevant_doc_ids": ["corporate_report_en.pdf_0"]},
      {"query": "Margen EBITDA", "relevant_doc_ids": ["informe_corporativo_es.pdf_2"]}
    ],
    "embedding_model": "openai",
    "k": 5
  }'
```

---

## Metadata Filters

Filter by any indexed metadata field:

```json
{
  "query": "revenue",
  "filters": {
    "source_type": "pdf"
  }
}
```

```json
{
  "query": "units sold",
  "filters": {
    "source_type": "csv"
  }
}
```

```json
{
  "query": "margen",
  "filters": {
    "language": "es"
  }
}
```

Range filters:

```json
{
  "query": "profit",
  "filters": {
    "page": {"gte": 1, "lte": 3}
  }
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `Precision@K` | Fraction of retrieved docs that are relevant |
| `Recall@K` | Fraction of relevant docs that are retrieved |
| `MRR` | Mean Reciprocal Rank of first relevant result |
| `NDCG@K` | Normalized Discounted Cumulative Gain |
| `latency_ms` | End-to-end retrieval latency |
| `p95_latency_ms` | 95th percentile latency across benchmark queries |

---

## Embedding Models

### OpenAI `text-embedding-3-large`
- Dimension: 3072
- Strong multilingual performance
- Higher cost, lower latency on small batches

### Cohere `embed-v4.0`
- Dimension: 1024
- Native multilingual support
- Configured via `COHERE_BASE_URL` for custom endpoint
- Uses `input_type=search_query` for queries, `search_document` for indexing

---


