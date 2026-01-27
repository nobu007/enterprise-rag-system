# Feature & User Story Definition

本ドキュメントは、`EPIC_PLANNING.md` で定義されたエピックに基づき、具体的な開発単位である **Feature** と、それを実装するための **User Story** を定義します。

---

## 🎯 Feature 1: テスト自動化基盤の構築 (Week 0)

### 概要 (Overview)
`pytest` を用いた自動テスト環境を構築し、GitHub Actions でのCIパイプラインを整備する。
開発者が安心してコードを変更できる状態（リグレッション防止）を作り、品質担保の土台とする。

### 親 Epic
- **Epic 1:** 技術基盤の適正化 (Foundation Refactoring)
- **寄与:** テスト容易性を向上させ、今後の機能追加時のバグ混入リスクを低減する。

### ユーザー価値 (User Value)
- **開発チーム:** 手動テストの手間を減らし、デグレードを早期に検知できる。
- **エンドユーザー:** バグの少ない安定したシステムを利用できる。

### 機能スコープ (Functional Scope)
**In Scope:**
- `pytest` 設定 (`pytest.ini`, `conftest.py`)
- 主要サービス (`rag_pipeline`, `retrieval`) の単体テスト
- 統合テスト (`tests/integration`) の基盤作成
- GitHub Actions workflow (`test.yml`) の修正

**Out of Scope:**
- UI (Streamlit) の自動テスト (E2Eテスト)
- 全モジュールのカバレッジ100%達成 (まずは主要パスのみ)

### 受け入れ条件 (Acceptance Criteria)
- [ ] `pytest` コマンド一発で全テストが実行可能であること
- [ ] CI (GitHub Actions) 上でテストがパスすること
- [ ] `app/services/` 配下の主要ロジックのカバレッジが 60% を超えること

### 見積もり (Estimation)
- **Total Effort:** 8h
- **Sprints:** 0.5 Sprint (Week 0)

### User Stories Decomposition

#### Story 1.1: pytest環境の設定とUnit Test作成
*   **As a** 開発者
*   **I want** ローカルで簡単にテストを実行できる環境が欲しい
*   **So that** 開発中に自分の変更が他の機能を壊していないか即座に確認するため
*   **Tasks:**
    *   [ ] `pytest.ini` を作成し、パスやオプションを設定する
    *   [ ] `tests/unit/test_rag_pipeline.py` を実装する (Mock利用)
    *   [ ] `tests/conftest.py` に共通フィクスチャ (Mock DB等) を定義する

#### Story 1.2: CIパイプラインの整備
*   **As a** 開発者
*   **I want** PR作成時に自動でテストが走るようにしたい
*   **So that** バグのあるコードがメインブランチにマージされるのを防ぐため
*   **Tasks:**
    *   [ ] `.github/workflows/test.yml` を修正し、依存関係インストールとpytest実行を記述
    *   [ ] テスト失敗時にPRのマージをブロックする設定を確認

---

## 🎯 Feature 2: 高度なドキュメント取り込みパイプライン (Week 1)

### 概要 (Overview)
`Unstructured.io` 等を用いて、PDF, Word, PowerPoint などの非構造化データからテキスト、表、画像を抽出し、RAG用の知識として利用可能にする。

### 親 Epic
- **Epic 2:** 検索・RAG精度の高度化 (Advanced RAG Core)
- **寄与:** 検索対象となるデータの質と量を向上させ、回答精度を高める。

### ユーザー価値 (User Value)
- **ナレッジ管理者:** 既存の社内ドキュメント(PDF等)をそのままの形式でアップロードできる。
- **ユーザー:** 図表に含まれる情報についても検索・回答が得られる。

### 機能スコープ (Functional Scope)
**In Scope:**
- マルチフォーマット対応 (PDF, DOCX, PPTX)
- 表 (Table) データの構造化抽出
- セマンティックチャンキング (意味の切れ目での分割)

**Out of Scope:**
- OCR (画像内文字認識) のチューニング (基本機能のみ利用)
- 音声・動画ファイルの対応

### 受け入れ条件 (Acceptance Criteria)
- [ ] PDF, DOCX, PPTX ファイルをエラーなくテキスト化できること
- [ ] ドキュメント内の表が崩れずにMarkdownまたはHTMLとして抽出されること
- [ ] チャンク分割が文の途中で行われていないこと

### 見積もり (Estimation)
- **Total Effort:** 40h
- **Sprints:** 1 Sprint (Week 1)

### User Stories Decomposition

#### Story 2.1: Unstructured.io の統合と検証
*   **As a** バックエンドエンジニア
*   **I want** 様々なファイル形式からテキストを抽出するライブラリを組み込みたい
*   **So that** アップロードされたファイルをRAGで扱えるテキスト形式に変換するため
*   **Tasks:**
    *   [ ] `requirements.txt` に `unstructured` 関連ライブラリを追加
    *   [ ] `DocumentLoader` サービスにファイル形式ごとのロード処理を実装
    *   [ ] サンプルPDFを用いて抽出精度を検証する

#### Story 2.2: セマンティックチャンキングの実装
*   **As a** ユーザー
*   **I want** 文脈が途切れないように文章が分割されていること
*   **So that** 検索時に意味の通った情報がヒットするようにするため
*   **Tasks:**
    *   [ ] LangChainの `SemanticChunker` または再帰的分割ロジックを実装
    *   [ ] チャンクサイズとオーバーラップのパラメータを調整
    *   [ ] メタデータ（ページ番号、ファイル名）を各チャンクに付与する

#### Story 2.3: 表データの構造化抽出
*   **As a** ユーザー
*   **I want** ドキュメント内の表データも正しく検索対象にしたい
*   **So that** 数値データやマトリクス情報を含む質問に答えられるようにするため
*   **Tasks:**
    *   [ ] PDF/Docx内のTable要素を検出し、Markdown形式等で抽出する処理を追加
    *   [ ] 表データの説明文（キャプション）をチャンクに含める処理
