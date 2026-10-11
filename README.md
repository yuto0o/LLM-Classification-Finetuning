# LLM Classification Finetuning

Kaggleの[LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning/)コンペティション用のコードです。

## セットアップ

### 1. リポジトリを取得する

```bash
git clone <このリポジトリのURL>
cd kaggle
uv sync
```

### 2. Kaggleの認証情報を設定する（初回のみ）

このコンペティションへの参加登録とチームへの参加を済ませたうえで、次を実行します。

```bash
uv run python src/kaggle/utils/kaggle_api/kaggle_login.py
```

表示された案内に従ってKaggleへのログインを完了してください。

すでにKaggleの認証情報を設定済みの環境では、この手順は不要です。認証情報が未設定の場合や、別のKaggleアカウントに切り替える場合だけ実行してください。

### 3. データをダウンロードする

```bash
uv run python src/kaggle/utils/kaggle_api/data_download.py
```

データは `src/kaggle/utils/raw_data/` に保存されます。`raw_data/` の中身はGit管理対象外なので、各メンバーが自分の環境でダウンロードしてください。

`src/kaggle/utils/raw_data/` はclone時には作成されない場合があります。ダウンロードスクリプトが実行時に必要なディレクトリを自動作成します。

## ディレクトリ構成

```text
src/kaggle/
├── models/             # モデルと学習コード
└── utils/
	├── kaggle_api/     # Kaggleへのログイン・データ取得
	└── raw_data/       # ダウンロードしたデータ（Git管理対象外）
```
