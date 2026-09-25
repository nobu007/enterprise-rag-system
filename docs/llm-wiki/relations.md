# Repository relations
Repository: enterprise-rag-system

No evidenced relationships yet.

- Observation: the kept data-integration spine is self-contained (FastAPI +
  local FAISS + optional Pinecone/OpenAI SDKs consumed as libraries); the
  2026-09-23 ingest log-sanitisation run discovered no cross-repo imports,
  CLI calls, or generated-artifact contracts to record.

## 階層関係（エスカレーション経路）

- 親: business_operation_notes（jinno確定 2026-09-26）
- 根拠: 特定事業の商品でない汎用データ統合スパイン
- 出典: contracts registry `registry/organization/repositories/enterprise-rag-system.yaml` の spec.parent（contracts commit ecbc226）。2026-09-26 の一括レビュー表（/home/jinno/output/repo-parent-review-2026-09-26.md）を jinno が現状案で承認。

