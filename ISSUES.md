# 改善提案 (Issues)

> **2026-09-13 スリムダウン後の正**: 本リポジトリはデータ統合スパイン（ローダ・
> パーサ・チャンク分割・バリデーション・埋め込み・ベクトルストア + 最小 Ingest API）
> に整理された。それ以外の機能（query/パイプライン/リランキング/キャッシュ/Celery
> バッチ/暗号化/テナント/レート制限/メトリクス等）は削除済みで、削除対象を題材に
> した旧 Issue は完了または無効化した。新しい Loop 候補は本ファイルの先頭未完了項目。

---

## Issue 1: テスト戦略の確立と実装 — **完了・2026-09-13 再確認**

- [x] `pytest` の設定ファイル (`pytest.ini`) を作成する — `pytest.ini` 実在（[pytest]・test_*.py discovery・asyncio auto）
- [x] 単体テスト — 2026-09-13 スリムダウン後: **260 passed / 0 failed**（`.venv310`）
- [x] テスト実行用のドキュメントを更新する — README「Tests」節に `.venv310/bin/python -m pytest tests/ -q` を記載
- 注: 旧タスクの `rag_pipeline` / `query.py` テストは対象モジュール削除に伴い閉じた

## Issue 2: オブザーバビリティの向上 (構造化ロギング) — **完了**

- [x] ロギング設定を行うユーティリティモジュールを作成する — `app/core/logging_config.py`（`get_logger()`・request ID contextvars・サニタイズ）
- [x] `print()` をロガー呼び出しに置換する — kept モジュールから `print()` は消滅（2026-09-13 確認）
- 注: `RequestIDMiddleware` タスクはミドルウェア削除に伴い閉じた

## Issue 3: 非同期処理の最適化 — **無効化（対象削除）**

- 旧対象は `RAGPipeline` / `query.py`（2026-09-13 削除）。kept コードの `openai` クライアントは `embeddings.py` 内で `OpenAI`/`AsyncOpenAI` 両対応済み。

## Issue 4: セキュリティと構成管理の強化 — **完了**

- [x] `ALLOWED_ORIGINS` を settings 経由に — `app/core/config.py`
- [x] CORS の `"*"` 消滅 — `app/main.py` は `settings.ALLOWED_ORIGINS` / `ALLOWED_HEADERS_LIST` を参照
- [x] ハードコードパスの排除 — `FAISS_INDEX_PATH` 等は Settings Field に集約

## Issue 5: Dependency Injection の適正化 — **無効化（対象削除）**

- 旧対象の `_rag_pipeline` グローバル / `dependencies.get_rag_pipeline` は 2026-09-13 削除済み。現行 `main.py` は lifespan で `app.state` に初期化し、ルータは必要時に lazy import する。

---

## Issue 6: CLI ingest がバリデーションゲートを通らない

**内容:** `scripts/ingest.py` は `DocumentLoader.load_directory` の結果をそのまま
分割・埋め込みする一方、API 経路（`POST /api/v1/documents/ingest`）は
`DocumentValidator` で品質ゲート（空/短文・PII・XSS/SQLi パターン）を通す。
同じデータを CLI から入れるとゲート無しで流入し、経路間で挙動が不一致。

**タスク:**
- [ ] `scripts/ingest.py` に `DocumentValidator.validate_batch` を組み込み、無効ドキュメントをスキップして件数を報告する（API と同じゲート）

## Issue 7: `collection` 引数が ABC/Pinecone の `upsert` に存在しない

**内容:** `VectorDB` ABC と `PineconeVectorDB.upsert` のシグネチャは
`(vectors, ids, metadata)` のみで `collection` 引数がなく、`collection` を
受け付けるのは `FAISSVectorDB.upsert` だけ。そのため
- `scripts/ingest.py` は `collection=` を渡さず常に `"default"` に入る
- API 経路（`documents.py`）は `collection=` を渡すため、Pinecone バックエンドでは `TypeError` になる潜在バグ

という不整合が残っている（2026-09-13 のシグネチャ照会で確認）。

**タスク:**
- [ ] `VectorDB` ABC と `PineconeVectorDB.upsert` に `collection: str = "default"` を追加する（Pinecone は namespace への写像、非対応なら明示的なエラーまたは注記）
- [ ] `scripts/ingest.py` の `upsert` 呼び出しに `collection=args.collection` を渡す
