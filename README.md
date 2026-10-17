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

### 4. 学習用データを作成する（層別分割）

```bash
uv run python src/kaggle/models/v1/utils/data_split.py
```

`train.csv` を層別5分割し、fold 0 を検証（val）、fold 1 をテスト（test）、それ以外を学習に使うデータとして `src/kaggle/models/v1/data/` に保存します。ラベルは `winner_model_a / winner_model_b / winner_tie` のどれか1個だけが真であることを検証したうえで、0（Model A）/ 1（Model B）/ 2（引き分け）に変換します。

## 学習とTensorBoard

学習を開始するには、次を実行します。

```bash
uv run python src/kaggle/models/v1/train.py
```

学習ログは `src/kaggle/models/v1/tb_logs/` に保存されます。学習とは別のターミナルでTensorBoardを起動します。

```bash
uv run tensorboard --logdir src/kaggle/models/v1/tb_logs --port 6006
```

ブラウザで次のURLを開くと、`train_loss` や `val_loss` などを確認できます。

```text
http://localhost:6006
```

VS Code RemoteやSSH環境で実行している場合は、VS Codeのポート転送でポート `6006` を転送してから開いてください。TensorBoardを終了するには、起動したターミナルで `Ctrl+C` を押します。

### 監視する指標

この大会のスコアは **multi-class log loss** です。そのため `val_loss` は温度もラベルスムージングも適用していない素の交差エントロピーにして、大会のスコアと同じ値になるようにしています。最良チェックポイントの選択と早期打ち切りは `val_loss`（最小化）で行います。`val_acc` / `val_f1` は診断用の参考値です。

- `*_loss`: 大会指標と同じ log loss（監視するのはこれ）
- `*_loss_opt`: 実際に最適化している損失（label smoothing 込み）
- `*_acc` / `*_f1`: accuracy と macro F1（診断用）

### 設定の差し替え

ハイパーパラメータはコードを書き換えずにコマンドライン引数で変更できます（`--help` で一覧が出ます）。

```bash
# GPUを1枚に絞る（複数ある場合は --devices 0,1 のようにも書けます）
uv run python src/kaggle/models/v1/train.py --devices 2

# 短い実験（学習バッチを絞る）
uv run python src/kaggle/models/v1/train.py --max-steps 300

# 上位モデルへ上げる（VRAMが足りないときは gradient checkpointing）
uv run python src/kaggle/models/v1/train.py --model-name microsoft/deberta-v3-base --batch-size 4 --gradient-checkpointing
```

よく使う設定:

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `--model-name` | `microsoft/deberta-v3-small` | 事前学習モデル |
| `--max-length` | 512 | 系列長。DeBERTa-v3は相対位置埋め込みが512までなので超えないこと |
| `--prompt-ratio` | 0.15 | プロンプトに割り当てるトークン予算の割合（残りはA/Bで折半） |
| `--head-ratio` | 0.7 | 応答を切り詰めるとき先頭をどれだけ残すか（末尾は結論なので残す） |
| `--swap-augment-p` | 0.5 | A/Bを入れ替えラベルも反転させる拡張の確率（位置バイアスの対策） |
| `--batch-size` / `--accumulate-grad-batches` | 8 / 1 | 実効バッチサイズ = batch-size × accumulate |
| `--lr` / `--warmup-ratio` / `--grad-clip` | 1e-5 / 0.06 / 1.0 | 学習率と勾配の管理 |
| `--label-smoothing` | 0.05 | log lossの校正を兼ねる |
| `--epochs` / `--max-steps` / `--limit-train-batches` | 3 / -1 / 1.0 | 学習量 |
| `--devices` / `--precision` | auto / bf16-mixed | 実行系。CPUで動かすなら `--accelerator cpu --precision 32-true` |

## 提出ファイルの作成

```bash
uv run python src/kaggle/models/v1/predict.py
```

`src/kaggle/models/v1/checkpoints/` 内で `val_loss` が最小のチェックポイントを使い、`raw_data/test.csv` に対して `id, winner_model_a, winner_model_b, winner_tie` の形式で `src/kaggle/models/v1/submissions/submission.csv` を書き出します。

- `--temperature auto`（既定）: 検証集の log loss を最小化する温度を探して確率を校正します。数値を直接指定することもできます。
- `--swap-tta`（既定ON）: A/Bを入れ替えた予測も平均して、提示順による偏りを減らします。`--no-swap-tta` で無効化できます。
- `--source val` / `--source test`: 分割済みデータで推論し、局所の log loss / accuracy / macro F1 を表示します。

## 動作確認（スモークテスト）

合成データで「前処理 → 分割 → 1ステップの学習 → 提出ファイル生成」までをつなげて確認します。数分程度で終わります（CPUで動かし、モデルの重みをダウンロードします）。

```bash
uv run python scripts/smoke_test.py
```

## 入力の前処理について

`prompt` / `response_a` / `response_b` を1系列にまとめますが、合計が512トークンを超えることが多いため、末尾を切り詰めると `response_b` が丸ごと失われます。ここではパートごとにトークン予算を割り振り（`--prompt-ratio`）、各パートを head/tail に切り詰めます（`--head-ratio`）。実際の長さ分布は次で確認できます。

```bash
uv run python src/kaggle/models/v1/utils/analyze/prompt_max_length.py
```

## ディレクトリ構成

```text
src/kaggle/
├── models/
│   └── v1/
│       ├── config.py                 # CLIから差し替える設定
│       ├── train.py                  # 学習
│       ├── predict.py                # 提出ファイルの生成
│       ├── generator/                # Lightning モジュール
│       ├── data/                     # 分割済みデータ（Git管理対象外）
│       ├── checkpoints/              # 最良チェックポイントと train_config.json
│       ├── submissions/              # submission.csv
│       ├── tb_logs/                  # TensorBoardのログ
│       └── utils/
│           ├── data_split.py         # 読み込み・ラベル化・層別分割
│           ├── type.py               # Dataset と前処理
│           └── analyze/              # 長さ分布などの解析
└── utils/
    ├── kaggle_api/                   # Kaggleへのログイン・データ取得
    └── raw_data/                     # ダウンロードしたデータ（Git管理対象外）
```

