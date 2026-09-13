# Enterprise RAG System

Document ingestion pipeline for retrieval-augmented generation: load,
validate, chunk, embed, and store documents into a vector database, with a
small FastAPI surface for ingestion and statistics.

**Scope note (2026-09-13 slim-down):** this repository was reduced to the
data-integration spine — loading, PDF/table parsing, chunking, validation
(PII / XSS / SQL-injection / quality checks), embedding, and vector
storage. LLM answer generation and query serving, response caching,
background batch queueing, request throttling, tenancy, encryption,
metrics, and the demo UI were removed. See `PURPOSE.md`.

## Features

- **Loaders** — `.txt` / `.md` / `.pdf` from single files or directories,
  with page-level PDF extraction and content-hash document IDs.
- **Parser** — text extraction with optional `pdfplumber` table extraction
  (CSV / markdown / JSON table formats).
- **Chunking** — `TextSplitter` with configurable size / overlap and a
  fixed-size fallback for separator-free text (e.g. CJK).
- **Validation** — pre-ingestion quality gate: PII patterns, XSS / SQL
  injection patterns, empty/short content, encoding checks; invalid
  documents are rejected with per-document error reports.
- **Embeddings** — OpenAI (`text-embedding-*`) and Cohere backends.
- **Vector storage** — FAISS (local, multi-collection, persisted to disk)
  and Pinecone (managed) backends behind a common interface.
- **Ingestion API** — directory ingest, single-file upload, statistics.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/documents/ingest` | Ingest a directory of documents |
| POST | `/api/v1/documents/upload` | Upload and ingest a single file |
| GET | `/api/v1/documents/stats` | Document / collection statistics |
| GET | `/health`, `/health/detailed` | Health checks |

Interactive docs: `/docs` (Swagger UI), `/redoc`, `/openapi.json`.

## Requirements

- Python 3.10 (the test environment is `.venv310`)
- `OPENAI_API_KEY` (required; see `.env.example` / environment)

## Install

```bash
python3.10 -m venv .venv310
.venv310/bin/pip install -r requirements.txt
```

## Run

```bash
.venv310/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Ingest from the CLI

```bash
.venv310/bin/python scripts/ingest.py --source ./docs --collection my-docs
```

`scripts/ingest.py` wires the same pipeline as the API (loader → splitter →
embeddings → vector store) and supports `--db-type faiss|pinecone`.

## Ingest via the API

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_path": "./docs", "collection": "my-docs", "chunk_size": 1000, "chunk_overlap": 200}'
```

## Configuration

All settings are environment variables (or `.env` entries); see
`app/core/config.py`. Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | (required) | OpenAI embeddings |
| `COHERE_API_KEY` | – | Cohere embeddings backend |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `FAISS_INDEX_PATH` | `./data/faiss_index.bin` | Local index file |
| `PINECONE_API_KEY` / `PINECONE_ENVIRONMENT` / `PINECONE_INDEX_NAME` | – | Pinecone backend |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | HTTP server |
| `ALLOWED_ORIGINS` / `ALLOWED_HEADERS` | localhost set | CORS |

## Layout

```
app/
  api/routes/     # documents (ingest/upload/stats), health
  core/           # config, embeddings, vectordb, logging
  services/       # document_loader, document_loader_enhanced, parser, validator
scripts/ingest.py # CLI ingestion
tests/            # pytest suite
```

## Tests

```bash
.venv310/bin/python -m pytest tests/ -q
```

## License

MIT
