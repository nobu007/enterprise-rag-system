# Repository relations
Repository: enterprise-rag-system

No evidenced relationships yet.

- Observation: the kept data-integration spine is self-contained (FastAPI +
  local FAISS + optional Pinecone/OpenAI SDKs consumed as libraries); the
  2026-09-23 ingest log-sanitisation run discovered no cross-repo imports,
  CLI calls, or generated-artifact contracts to record.

## 階層関係（エスカレーション経路）

- 親: business_operation_notes（推定・jinno確定待ち）
- 根拠: RAG 前処理の汎用データ統合スパイン（特定事業の商品でなく共通インフラ的実装）であり、既存の relations 記録でも自己完結とされている。
- 出典: README.md の Scope note（2026-09-13 slim-down）、docs/llm-wiki/relations.md、原則 llm-wiki-discipline の drafts-are-status-marked（状態表示付き草案）。

- Observation: 上記の親は推定草案であり、jinno確定後に contracts registry の spec.parent へ反映される。provider/consumer の検証済み関係はまだない。
