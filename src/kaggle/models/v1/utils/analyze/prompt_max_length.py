"""入力の長さ分布を文字数とトークン数の両方から見る解析スクリプト.

文字数だけ見ると「512 token に収まるか」が分からない。ここでは
train.py が使うトークン予算（prompt_ratio / head_ratio）で
実際に何トークン捨てられるかも表示する。

例::

    uv run python src/kaggle/models/v1/utils/analyze/prompt_max_length.py
    uv run python src/kaggle/models/v1/utils/analyze/prompt_max_length.py --sample 0   # 全件
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from kaggle.models.v1.config import DEFAULT_MODEL_NAME
from kaggle.models.v1.utils.data_split import INPUT_COLUMNS
from kaggle.models.v1.utils.type import PART_PREFIXES, token_budgets, truncate_ids
from kaggle.utils.constants import DataPaths


def describe(values: list[int]) -> str:
    array = np.array(values, dtype=np.int64)
    quantiles = np.quantile(array, [0.5, 0.9, 0.95, 0.99])
    return (
        f"mean={array.mean():.1f} p50={quantiles[0]:.0f} p90={quantiles[1]:.0f} "
        f"p95={quantiles[2]:.0f} p99={quantiles[3]:.0f} max={array.max()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="prompt_max_length")
    parser.add_argument("--sample", type=int, default=2000, help="0 で全件")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prompt-ratio", type=float, default=0.15)
    parser.add_argument("--head-ratio", type=float, default=0.7)
    args = parser.parse_args()

    dataset = pd.read_csv(DataPaths.RAW_DATA_DIR / "train.csv")
    if args.sample and len(dataset) > args.sample:
        dataset = dataset.sample(args.sample, random_state=0)

    print(f"rows analyzed: {len(dataset)}")
    for column in ("prompt", "response_a", "response_b"):
        print(
            f"[chars] {column}: {describe(dataset[column].astype(str).str.len().tolist())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt_budget, response_a_budget, response_b_budget = token_budgets(
        max_length=args.max_length, prompt_ratio=args.prompt_ratio
    )
    print(
        f"\ntoken budget (max_length={args.max_length}, prompt_ratio={args.prompt_ratio}): "
        f"prompt={prompt_budget} response_a={response_a_budget} response_b={response_b_budget}"
    )

    budgets = {
        "prompt": (prompt_budget, 1.0),
        "response_a": (response_a_budget, args.head_ratio),
        "response_b": (response_b_budget, args.head_ratio),
    }
    for column in ("prompt", "response_a", "response_b"):
        budget, head_ratio = budgets[column]
        lengths = []
        dropped = []
        for text in dataset[column].astype(str):
            ids = tokenizer.encode(
                PART_PREFIXES[column] + text, add_special_tokens=False
            )
            lengths.append(len(ids))
            dropped.append(max(0, len(ids) - budget))
        dropped_array = [value for value in dropped if value > 0]
        print(
            f"[tokens] {column}: {describe(lengths)} / "
            f"切り詰められる行: {len(dropped_array) / len(dataset):.1%} "
            f"(平均 {sum(dropped_array) / max(1, len(dropped_array)):.0f} token, "
            f"head_ratio={head_ratio})"
        )

    total = [
        sum(
            len(
                truncate_ids(
                    tokenizer.encode(
                        PART_PREFIXES[c] + str(row[c]), add_special_tokens=False
                    ),
                    budget=budget,
                    head_ratio=ratio,
                )
            )
            for c, (budget, ratio) in budgets.items()
        )
        for row in dataset[INPUT_COLUMNS].to_dict(orient="records")
    ]
    print(f"\n[組み立て後の系列長] {describe(total)} (max_length={args.max_length})")


if __name__ == "__main__":
    main()
