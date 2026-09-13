# 改善提案 (Issues)

> **2026-09-13 スリムダウン後の正**: 本リポジトリはデータ統合スパイン（ローダ・
> パーサ・チャンク分割・バリデーション・埋め込み・ベクトルストア + 最小 Ingest API）
> に整理された。それ以外の機能（query/パイプライン/リランキング/キャッシュ/Celery
> バッチ/暗号化/テナント/レート制限/メトリクス等）は削除済みで、削除対象を題材に
> した旧 Issue は完了または無効化した。新しい Loop 候補は本ファイルの先頭未完了項目。

---

## Issue 12: vectordb の残存整理 — 生 `collection` ログと rebuild 後のレガシー別名 — **完了・2026-09-14**

**内容:** Issue 11 の `delete()` 成功ログは `sanitize_for_log` で統一したが、
`app/core/vectordb.py` にはクライアント指定の `collection` を生のまま埋め込む
ログが残っている（CWE-117 同種。search / delete の not-found 経路は対応済み）。
また `_rebuild_without_ids` は `self.indices[collection]` を新辞書に差し替えるため、
`__init__` / `connect` が張るレガシー別名が rebuild 後に陳腐化する
（2026-09-14 の外部読み手 grep で現行読み手ゼロを確認済み・潜在バグ）。

**タスク:**
- [x] `_create_collection_index` の "Created FAISS index for collection" info（:265）
- [x] `upsert` の "Upserted ... into collection" info（:474）
- [x] `save` の "Saved FAISS index for collection" info ×2（:621, :643）—
      メッセージの `index_path` も `f"{path}.{collection}"` で collection を
      埋め込むため併せてサニタイズ
- [x] rebuild 後に `self.id_to_idx` / `self.idx_to_id` 別名を貼り直すか、読み手のない
      レガシー別名を廃止する → 貼り直しを採用（diff 最小・外部互換維持）
- ✅ 2026-09-14 run: 生 `collection` ログ 4 箇所を `sanitize_for_log` 経由に変更
  （save は `index_path` 側も）。`_rebuild_without_ids` が default のレガシー別名
  `self.id_to_idx` / `self.idx_to_id` を新辞書へ再張り（pre-fix で
  `{'doc1': 0, 'doc2': 1}` の陳腐化をピンテストが再現）。create/upsert/save の
  CRLF ログ注入ピンテストと別名再張りピンテストを追加（pre-fix 失敗確認済み）。
  **297 passed / 0 failed**・`compileall app` OK。

## Issue 11: upsert 同一バッチ内の重複 ID が二重登録される／`delete()` が未対応のまま — **完了・2026-09-14**

**内容:** Issue 10 の修正（`6a00e77`）はインデックス既存 ID との照合のみで、
同一 upsert 呼び出し内の重複（`ids=["x", "x"]`。id が既知でも rebuild 後に
バッチ側の 2 件がそのまま追加される）は崩れていない。また `delete()` は
"not supported" 警告のままで、`_rebuild_without_ids` と同じ機構で実装可能。

**タスク:**
- [x] upsert 冒頭でバッチ内重複も除去する（末尾勝ち。既知 ID 照合の前段で ids を set 化して len 比較）
- [x] `FAISSVectorDB.delete()` を `_rebuild_without_ids` に委譲し "not supported" 警告を解消する
- [x] ピンテスト: `test_upsert_duplicate_ids_within_one_batch_stay_unique` を
  `tests/unit/test_vectordb_collections.py` に strict xfail で追加済み。修正後に
  XPASS でスイートが失敗するため、その際はマーカーを外すこと
- ✅ 2026-09-14 run: upsert 冒頭でバッチ内重複を末尾勝ちで畳み込み（既知 ID 照合の前段）、
  `delete()` を `_rebuild_without_ids` に委譲（未知 ID・未知コレクションは no-op）。
  rebuild 時に削除 ID のメタデータも `metadata_stores` から除去。xfail マーカーを外し、
  delete のピンテスト `test_delete_drops_id_and_keeps_collection_searchable` を追加。
  **288 passed / 0 failed**・`compileall app` OK。

## Issue 10: FAISS `upsert` が追記専用で同一 ID が重複する — **完了・2026-09-14**

**内容:** `FAISSVectorDB.upsert` は ABC 契約（"Insert or update vectors"）に反して
常に `index.add()` で追記しており、ドキュメント ID はコンテンツハッシュのため
再 ingest（README 記載の CLI 再実行を含む）で同一 ID のベクトルが蓄積し、
`search` が同一ドキュメントを古いコピーの数だけ重複返却し `get_stats` も膨張する
（再現: 再 upsert 後 `ntotal` 1→2、hits `['doc-1', 'doc-1']`）。

**タスク:**
- [x] 既存 ID を含む upsert で旧ベクトルを排除してから追加する（flat index は
      in-place 削除不可のため `_rebuild_without_ids` で同一メトリックの新 index へ再構築）
- [x] 再 upsert が重複を生まないこと・混在バッチ・L2 メトリック保存・save→connect
      往復を `tests/unit/test_vectordb_collections.py` に追記してピン留め
- ✅ 2026-09-14 run: **284 passed / 0 failed**（280 既存 + 4 新規）、`compileall app` OK

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

## Issue 9: `connect()` 後の default コレクションへの upsert が KeyError・search が無音に 0 件 — **完了・2026-09-14**

**内容:** `FAISSVectorDB.connect()` の JSON メタデータ読み込み分支が
`metadata_store` 等のレガシー属性にのみ格納し、pickle 分支
（`metadata_stores["default"]` 等を初期化）と異なりコレクションごとの
辞書に格納していなかった。ディスク上の既存インデックスに `connect()` した
fresh インスタンスでは

- `upsert(collection="default")` → `"default"` は `self.indices` に存在するため
  `_get_or_create_collection` のブートストラップが skip され `KeyError: 'default'`
- `search(collection="default")` → `idx_to_id` が空のため全ヒットが無音に棄却される（サイレントデータロス）

という不整合が残っていた（2026-09-14 の再現スクリプトで確認。`default` は
ingest API が collection 未指定時に書き込む先で、lifespan・`/stats` が
`connect()` を呼ぶため現実の経路）。

**タスク:**
- [x] JSON 分支にも pickle 分支と同じ 3 行（`metadata_stores` / `id_to_idx_mappings` / `idx_to_id_mappings` への格納）を追加する
- [x] default コレクションの save → connect ラウンドトリップ後の upsert・search（metadata/text 含む）を検証する回帰テストを追加する
- ✅ 2026-09-14 run: `app/core/vectordb.py` 修正 + `tests/unit/test_vectordb_collections.py` に `test_vector_db_default_collection_persistence` 追加。修正前は当該テストが `KeyError: 'default'` で失敗することを確認済み。**278 passed / 0 failed**・`compileall` クリア。
