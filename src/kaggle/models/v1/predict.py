"""学習済みチェックポイントから提出ファイル（submission.csv）を作るスクリプト.

この大会の提出は `id, winner_model_a, winner_model_b, winner_tie` の確率列で、
スコアは multi-class log loss である。そのため

- 温度スケーリングで確率を校正する（`--temperature auto` は検証集で探索する）
- A/B を入れ替えた予測も平均する（swap TTA。位置バイアスを減らす）

例::

    uv run python src/kaggle/models/v1/predict.py                       # 最良 ckpt + 検証集で温度を決める
    uv run python src/kaggle/models/v1/predict.py --temperature 1.2 --no-swap-tta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from kaggle.models.v1.config import TARGET_COLUMNS, TrainConfig
from kaggle.models.v1.generator.deberta_generator import ChatLightningModule
from kaggle.models.v1.utils.data_split import load_chat_frame
from kaggle.models.v1.utils.type import build_encoding, make_collate_fn
from kaggle.utils.constants import DataPaths, ModelPaths

# log loss が inf にならないように確率を丸める範囲
PROBABILITY_CLIP = 1e-6
TEMPERATURE_GRID = np.linspace(0.5, 4.0, 72)


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """ModelCheckpoint が残した .ckpt を 1 つ選ぶ（複数あれば val_loss が小さいもの）."""
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"{checkpoint_dir} にチェックポイントがありません。先に train.py を実行してください。"
        )
    if len(candidates) == 1:
        return candidates[0]

    def val_loss_from_name(path: Path) -> float:
        # filename="best-...-valloss{val_loss:.4f}" の形式から取り出す
        parts = path.stem.split("valloss")
        try:
            return float(parts[-1])
        except ValueError:
            return float("inf")

    best = min(candidates, key=val_loss_from_name)
    print(f"{len(candidates)} 個のチェックポイントから {best.name} を選びました")
    return best


def parse_temperature(value: str) -> float | None:
    if value.strip().lower() == "auto":
        return None
    temperature = float(value)
    if temperature <= 0:
        raise ValueError(f"temperature は正の値にしてください: {temperature}")
    return temperature


def calibrate_temperature(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """検証集の log loss を最小化する温度を探す（温度スケーリングによる校正）."""
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    def logloss(temperature: float) -> float:
        return float(
            torch.nn.functional.cross_entropy(
                logits_tensor / temperature, labels_tensor
            )
        )

    before = logloss(1.0)
    best_temperature, best_value = 1.0, before
    for temperature in TEMPERATURE_GRID:
        value = logloss(float(temperature))
        if value < best_value:
            best_temperature, best_value = float(temperature), value
    return best_temperature, before, best_value


def _predict_logits(
    model: ChatLightningModule,
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
    *,
    swap: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """与えた順序（swap=True なら A/B を入れ替えた順序）で推論した logits を返す."""
    collate_fn = make_collate_fn(tokenizer)
    model.eval()
    chunks = []

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            items = []
            for row in chunk.itertuples(index=False):
                response_a, response_b = row.response_a, row.response_b
                if swap:
                    response_a, response_b = response_b, response_a
                items.append(
                    build_encoding(
                        tokenizer,
                        prompt=str(row.prompt),
                        response_a=str(response_a),
                        response_b=str(response_b),
                        max_length=cfg.max_length,
                        prompt_ratio=cfg.prompt_ratio,
                        head_ratio=cfg.head_ratio,
                    )
                )
            batch = {key: value.to(device) for key, value in collate_fn(items).items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=None,
            )
            chunks.append(outputs.logits.float().cpu().numpy())

    return np.concatenate(chunks, axis=0)


def predict_probabilities(
    model: ChatLightningModule,
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
    *,
    temperature: float,
    swap_tta: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """提出用の確率。swap した側の予測は A/B を元に戻してから平均する（tie はそのまま）."""
    logits = _predict_logits(
        model, df, tokenizer, cfg, swap=False, batch_size=batch_size, device=device
    )
    probabilities = torch.softmax(torch.tensor(logits) / temperature, dim=-1).numpy()

    if swap_tta:
        swapped_logits = _predict_logits(
            model, df, tokenizer, cfg, swap=True, batch_size=batch_size, device=device
        )
        swapped = torch.softmax(
            torch.tensor(swapped_logits) / temperature, dim=-1
        ).numpy()
        # swap した世界での (A, B, tie) を元の順序へ写し戻す
        swapped = swapped[:, [1, 0, 2]]
        probabilities = (probabilities + swapped) / 2.0

    return probabilities


def write_submission(df: pd.DataFrame, probabilities: np.ndarray, output: Path) -> None:
    probabilities = np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    submission = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            TARGET_COLUMNS[0]: probabilities[:, 0],
            TARGET_COLUMNS[1]: probabilities[:, 1],
            TARGET_COLUMNS[2]: probabilities[:, 2],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"提出ファイルを書きました: {output} ({len(submission)} 行)")

    predicted = probabilities.argmax(axis=1)
    distribution = pd.Series(predicted).value_counts(normalize=True).sort_index()
    labels = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
    print("予測クラスの分布:")
    for index, ratio in distribution.items():
        print(f"  {labels[index]}: {ratio:.4f}")


def evaluate(df: pd.DataFrame, probabilities: np.ndarray) -> None:
    """ラベルがあるソース（val / test）では局所スコアを表示する."""
    labels = df["label"].to_numpy()
    probabilities = np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    logloss = -np.log(np.take_along_axis(probabilities, labels[:, None], axis=1)).mean()
    accuracy = (probabilities.argmax(axis=1) == labels).mean()
    macro_f1 = f1_score(labels, probabilities.argmax(axis=1), average="macro")
    print(
        f"local logloss: {logloss:.6f}  accuracy: {accuracy:.4f}  macro F1: {macro_f1:.4f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="predict")
    parser.add_argument("--version", default="v1", help="models/<version> を使う")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="ckpt のパス（省略時は checkpoints 内で val_loss が最小のもの）",
    )
    parser.add_argument(
        "--source",
        choices=["competition", "val", "test"],
        default="competition",
        help="competition: raw_data/test.csv（ラベルなし）, val/test: 分割済みデータ",
    )
    parser.add_argument(
        "--temperature",
        default="auto",
        help="温度スケーリング。auto にすると検証集の log loss を最小化する値を探す",
    )
    parser.add_argument(
        "--swap-tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A/B を入れ替えた予測も平均する（位置バイアスの低減）",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="推論に使うデバイス（auto: cuda があれば cuda）",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="submission.csv の出力先"
    )
    args = parser.parse_args(argv)

    checkpoint_dir = ModelPaths.BASE_DIR / args.version / "checkpoints"
    data_dir = ModelPaths.BASE_DIR / args.version / "data"
    checkpoint = args.checkpoint or find_best_checkpoint(checkpoint_dir)
    output = (
        args.output
        or ModelPaths.BASE_DIR / args.version / "submissions" / "submission.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChatLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    cfg = model.cfg

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if args.source == "competition":
        source_path = DataPaths.RAW_DATA_DIR / "test.csv"
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} が見つかりません。kaggle_api/data_download.py でダウンロードしてください。"
            )
        df = load_chat_frame(source_path, with_label=False)
    else:
        df = load_chat_frame(data_dir / f"{args.source}_split.csv", with_label=True)

    temperature = parse_temperature(args.temperature)
    if temperature is None:
        if args.source == "competition":
            print(
                "temperature=auto のときは --source val/test を指定してください。1.0 を使います。"
            )
            temperature = 1.0
        else:
            val_logits = _predict_logits(
                model,
                df,
                tokenizer,
                cfg,
                swap=False,
                batch_size=args.batch_size,
                device=device,
            )
            temperature, before, after = calibrate_temperature(
                val_logits, df["label"].to_numpy()
            )
            print(
                f"温度を校正しました: temperature={temperature:.2f} "
                f"(val logloss {before:.6f} -> {after:.6f})"
            )

    print(f"checkpoint: {checkpoint}")
    print(
        f"model: {cfg.model_name} / max_length={cfg.max_length} / temperature={temperature:.2f}"
    )
    print(f"source: {args.source} ({len(df)} 行) / swap_tta={args.swap_tta}")

    probabilities = predict_probabilities(
        model,
        df,
        tokenizer,
        cfg,
        temperature=temperature,
        swap_tta=args.swap_tta,
        batch_size=args.batch_size,
        device=device,
    )

    if args.source != "competition":
        evaluate(df, probabilities)

    write_submission(df, probabilities, output)


if __name__ == "__main__":
    main()
