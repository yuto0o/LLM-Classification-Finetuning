"""決定的なデータで end-to-end の動作確認をするスクリプト.

合成データの 1 ステップ学習 → 最良チェックポイント → 提出ファイル生成までを
つなげて、前処理・メトリクス・提出形式が壊れていないことを確認する。

実行::

    uv run python scripts/smoke_test.py

`src/kaggle/models/smoke/` に一時ファイルを作り、最後に削除する。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaggle.models.v1 import predict as predict_module
from kaggle.models.v1 import train as train_module
from kaggle.models.v1.config import TARGET_COLUMNS, TrainConfig
from kaggle.models.v1.utils import data_split as data_split_module
from kaggle.models.v1.utils.type import (
    SPECIAL_TOKENS_TOTAL,
    ChatDataset,
    build_encoding,
    token_budgets,
    truncate_ids,
)
from kaggle.utils.constants import ModelPaths

SMOKE_VERSION = "smoke"
SMOKE_DIR = ModelPaths.BASE_DIR / SMOKE_VERSION

PROMPTS = [
    "Explain what a gradient is in one short paragraph.",
    "Write a haiku about the sea.",
    "How do I boil an egg?",
    "Summarize the plot of a detective story.",
]


def make_synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ラベルが偏らない合成データを作る（A/B/tie が均等になるようにする）."""
    records = []
    for index in range(24):
        label = index % 3
        long_text = "This is a long response. " * 40  # 切り詰めを検証するために長くする
        records.append(
            {
                "id": index,
                "prompt": PROMPTS[index % len(PROMPTS)],
                "response_a": f"Answer A for {index}. {long_text}",
                "response_b": f"Answer B for {index}. {long_text}",
                "winner_model_a": label == 0,
                "winner_model_b": label == 1,
                "winner_tie": label == 2,
            }
        )
    train_df = pd.DataFrame(records)

    test_records = [
        {
            "id": 1000 + index,
            "prompt": PROMPTS[index % len(PROMPTS)],
            "response_a": f"Test answer A {index}.",
            "response_b": f"Test answer B {index}.",
        }
        for index in range(6)
    ]
    return train_df, pd.DataFrame(test_records)


def check_preprocessing() -> None:
    from transformers import AutoTokenizer

    max_length = 64
    budgets = token_budgets(max_length=max_length, prompt_ratio=0.2)
    assert sum(budgets) == max_length - SPECIAL_TOKENS_TOTAL, budgets

    ids = list(range(100))
    cut = truncate_ids(ids, budget=10, head_ratio=0.7)
    assert len(cut) == 10
    assert cut[:7] == ids[:7] and cut[7:] == ids[-3:], cut
    assert truncate_ids(ids[:5], budget=10, head_ratio=0.7) == ids[:5]

    tokenizer = AutoTokenizer.from_pretrained(TrainConfig().model_name)
    encoding = build_encoding(
        tokenizer,
        prompt="a " * 500,
        response_a="b " * 2000,
        response_b="c " * 2000,
        max_length=max_length,
        prompt_ratio=0.2,
        head_ratio=0.7,
    )
    length = int(encoding["input_ids"].numel())
    assert length <= max_length, length
    # 3 パートすべてが系列に残っていること（末尾切り詰めだと response_b が消える）
    decoded = tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)
    assert "Model A:" in decoded and "Model B:" in decoded, decoded

    # swap 拡張: 学習時だけ A/B が入れ替わり、ラベルは 0<->1 だけ反転する
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "prompt": ["p"] * 3,
            "response_a": ["AAA"] * 3,
            "response_b": ["BBB"] * 3,
            "label": [0, 1, 2],
        }
    )
    dataset = ChatDataset(
        df, tokenizer, max_length=max_length, swap_augment_p=1.0, seed=7
    )
    dataset.set_epoch(0)
    for index, expected_label in enumerate([1, 0, 2]):
        item = dataset[index]
        decoded = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        assert "Model A: BBB" in decoded and "Model B: AAA" in decoded, decoded
        assert int(item["labels"]) == expected_label, (index, int(item["labels"]))
    dataset.set_epoch(1)
    assert int(dataset[0]["labels"]) == 1  # swap 確率 1.0 なので常に反転

    print("前処理のチェック: OK")


def main() -> None:
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    work_dir = SMOKE_DIR / "raw_data"
    work_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = make_synthetic_frames()
    train_path = work_dir / "train.csv"
    test_path = work_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    try:
        check_preprocessing()

        # test.csv のように target 列がない形式も読めること
        test_frame = data_split_module.load_chat_frame(test_path, with_label=False)
        assert len(test_frame) == len(test_df) and "label" not in test_frame.columns

        # フラグが 2 個立っている行は黙って Model A にせずエラーにする
        broken_path = work_dir / "broken.csv"
        broken = train_df.copy()
        broken.loc[0, "winner_model_b"] = True
        broken.to_csv(broken_path, index=False)
        try:
            data_split_module.load_chat_frame(broken_path, with_label=True)
        except ValueError as error:
            print(f"不正なラベル行を検知しました: {error}")
        else:
            raise AssertionError("不正なラベル行を検知できませんでした")

        data_split_module.main(
            [
                "--input",
                str(train_path),
                "--version",
                SMOKE_VERSION,
                "--n-splits",
                "3",
                "--seed",
                "0",
            ]
        )

        train_module.main(
            [
                "--version",
                SMOKE_VERSION,
                "--max-length",
                "64",
                "--batch-size",
                "2",
                "--eval-batch-size",
                "4",
                "--epochs",
                "1",
                "--warmup-ratio",
                "0.1",
                "--accelerator",
                "cpu",
                "--devices",
                "1",
                "--precision",
                "32-true",
                "--num-workers",
                "0",
            ]
        )

        checkpoint_dir = SMOKE_DIR / "checkpoints"
        assert list(checkpoint_dir.glob("*.ckpt")), (
            "チェックポイントが保存されていません"
        )
        saved_config = checkpoint_dir / "train_config.json"
        assert saved_config.exists(), "train_config.json が保存されていません"

        output = SMOKE_DIR / "submissions" / "submission.csv"
        predict_module.main(
            [
                "--version",
                SMOKE_VERSION,
                "--source",
                "val",
                "--temperature",
                "auto",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )

        submission = pd.read_csv(output)
        assert list(submission.columns) == ["id", *TARGET_COLUMNS], submission.columns
        val_df = pd.read_csv(SMOKE_DIR / "data" / "val_split.csv")
        assert len(submission) == len(val_df), (len(submission), len(val_df))
        probabilities = submission[TARGET_COLUMNS].to_numpy(dtype=float)
        assert (probabilities > 0).all() and (probabilities < 1).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)

        # 大会の sample_submission と同じ id 集合になること（test.csv から作る場合）
        predict_module.main(
            [
                "--version",
                SMOKE_VERSION,
                "--source",
                "test",
                "--temperature",
                "1.0",
                "--no-swap-tta",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--output",
                str(SMOKE_DIR / "submissions" / "submission_test.csv"),
            ]
        )
        print("\nsmoke test: OK")
    finally:
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
