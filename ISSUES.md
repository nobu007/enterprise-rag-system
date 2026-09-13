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

## Issue 6: CLI ingest がバリデーションゲートを通らない — **完了・2026-09-13**

- [x] `scripts/ingest.py` に `DocumentValidator().validate_batch` を組み込み、無効ドキュメントをスキップして件数を報告する（API と同じゲート）
- ✅ 2026-09-13 run: 実装は chain worktree の残骸から回収し、本体 checkout で検証のうえ取り込んだ（`tests/unit/test_ingest_script.py` 追加、**262 passed / 0 failed**）。全件無効時は `sys.exit(1)`。

## Issue 7: `collection` 引数が ABC/Pinecone の `upsert` に存在しない — **完了・2026-09-13**

**内容:** `VectorDB` ABC と `PineconeVectorDB.upsert` のシグネチャは
`(vectors, ids, metadata)` のみで `collection` 引数がなく、`collection` を
受け付けるのは `FAISSVectorDB.upsert` だけ。そのため
- `scripts/ingest.py` は `collection=` を渡さず常に `"default"` に入る
- API 経路（`documents.py`）は `collection=` を渡すため、Pinecone バックエンドでは `TypeError` になる潜在バグ

という不整合が残っている（2026-09-13 のシグネチャ照会で確認）。

**タスク:**
- [x] `VectorDB` ABC と `PineconeVectorDB.upsert` に `collection: str = "default"` を追加する（Pinecone は namespace への写像）
- [x] `scripts/ingest.py` の `upsert` 呼び出しに `collection=args.collection` を渡す
- ✅ 2026-09-13 run: Pinecone の upsert/search/delete を同じ namespace に揃え、`default` は既存の空 namespace を維持。CLI 引数伝播・101件のバッチ・省略時/明示 default/名前付き collection をモック検証。構文検証成功、**265 passed / 0 failed**（5 warnings）。

## Issue 8: ハーネスの PYTHONPATH が `scripts` を shadow してテスト収集が崩壊 — **完了・2026-09-14**

**内容:** `scripts/` に `__init__.py` が無い間は暗黙の namespace package になり、
sys.path 前方に外部の正規 `scripts` パッケージ（例: ハーネスが
`PYTHONPATH=/home/jinno/ai-hub` を export するが、その `scripts/` に `ingest.py`
は存在しない）が置かれると `import scripts.ingest` が `ModuleNotFoundError`
となり、テスト収集ごと exit 2 で崩壊していた。

**タスク:**
- [x] `scripts/__init__.py` を追加し、リポジトリローカルの正規パッケージが外部 `scripts` より優先されるようにする
- [x] `import scripts.ingest` が本リポジトリ配下で解決されることを保証する回帰テストを追加する
- ✅ 2026-09-14 run: 修正は `cbf193c` で着地済み（回帰テストを `tests/unit/test_ingest_script.py` に追加、README の CLI フラグ誤記 `--source-path`→`--source` も同時修正）。`PYTHONPATH=/home/jinno/ai-hub`・合成 shadow ツリー・unset の 3 条件で **266 passed / 0 failed** を確認。
