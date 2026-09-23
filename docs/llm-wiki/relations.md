# Repository relations
Repository: enterprise-rag-system

No evidenced relationships yet.

- Observation: the kept data-integration spine is self-contained (FastAPI +
  local FAISS + optional Pinecone/OpenAI SDKs consumed as libraries); the
  2026-09-23 ingest log-sanitisation run discovered no cross-repo imports,
  CLI calls, or generated-artifact contracts to record.
