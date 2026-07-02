# decisions.md — ACTIVE な AUTO 決定のキャッシュ（上限200行。消してよい）

> 本ファイルはキャッシュ。肥大化したら古い順に切り詰める（Rule 6）。

## 2026-07-02 bootstrap (cycle 0 / 初回生成)

### D-BOOT-001: bootstrap は ontology.yml を直接 seed した（Rule 1 との調整）
- **判断**: セクション6.4「初回生成完了後、Step 1(Discover) で ontology/invariants/mappings を充実させる」が
  より具体的な指示であり、本状況（.concept/ 未存在＝初回生成）に直接該当するため、これを優先。
- **Rule 1（新語は term_queue 経由）の扱い**: 以後のサイクルで発見される新規概念は term_queue.yml 経由とする。
  term_queue.yml は空のまま維持（bootstrap seed は例外）。
- **優先順位**: セクション0.1「本(13)と12で矛盾時は13を優先」にも合致。

### D-BOOT-002: 生成場所を worktree に置いた
- **判断**: 現在 git worktree（branch ai/enterprise-rag-system/instruction-*）で作業中のため、
  `.concept/` は worktree ルートに生成→branch に commit→main へマージで正規位置
  `/home/jinno/enterprise-rag-system/.concept/` に到達させる。
- spec の「常に /home/jinno/enterprise-rag-system 直下」はマージ後の最終配置として満たされる。

### D-BOOT-003: lint/test コマンドは CI を正とした
- **test**: `.github/workflows/test.yml` の `pytest tests/ -v --cov=app` を採用（pytest.ini の asyncio_mode=auto 使用）。
- **lint**: pyproject.toml/setup.cfg なし。requirements.txt に flake8==7.0.0 明示のため `flake8 app tests`。
- CI 上で lint は未実行（検出不能に近い）だが、依存に存在するため採用。

### D-BOOT-004: Reranker と QueryResultRanker は統合せず共存（AMB-001）
- 直交する概念（クロスエンコーダ精度向上 vs 複数特徴量の重み付け整列）。
- default: re-ranking 有効 / RANKING_ENABLED=False。conflicts ではなく ambiguities に記録。

### D-BOOT-005: README 言及の未検証機能は conflicts 空（保留）~~（→RESOLVED: cycle 1 で検証完了）~~
- ~~Weaviate/Chroma バックエンド、Document Relationship Graph 等の実装網羅度は
  後続サイクルで検証し、矛盾があれば conflicts.yml に記録。今サイクルでは断言しない（Rule 4）。~~
- **RESOLVED (cycle 1)**: 検証完了 → CFLT-VECTORDB-001(Weaviate), CFLT-DRG-001(Document Relationship Graph) として記録。Chroma は README の強い主張がないため CFLT-VECTORDB-001 に包含。

## 2026-07-02 maintenance (cycle 1 / README↔code 同期)

### AUTO:VectorDB.backend_coverage:implemented_only
- Status: ACTIVE
- Chosen: コード実装を正とし concrete バックエンドは Pinecone+FAISS のみ。Weaviate は未実装(aspirational)。
- Policy: code_is_truth + safety（実装根拠のない機能を概念として確定しない）
- Expires After Runs: 20
- Linked: CFLT-VECTORDB-001 / term VectorDB
- Revert Triggers: WeaviateVectorDB 実装追加、または README/docstring から Weaviate 削除

### AUTO:DocumentRelationshipGraph.feature_existence:unimplemented
- Status: ACTIVE
- Chosen: Document Relationship Graph は未実装(README のみ)。概念辞書に term 追加なし。
- Policy: code_is_truth + Rule 1/4（実装根拠なき新語・確定禁止）
- Expires After Runs: 20
- Linked: CFLT-DRG-001 / term Document
- Revert Triggers: DRG 実装追加(networkx + relationships route)、または README から DRG 削除
