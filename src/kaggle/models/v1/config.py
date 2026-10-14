"""学習・推論の設定をまとめるモジュール。

従来のようにコードを書き換えずに、コマンドライン引数から設定を差し替えられるようにする。

例::

    uv run python src/kaggle/models/v1/train.py --model-name microsoft/deberta-v3-base --epochs 5
    uv run python src/kaggle/models/v1/train.py --devices 2 --max-steps 50   # 自分のGPUで短時間だけ試す
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import get_type_hints

# 0: Model A の応答が好まれた / 1: Model B / 2: 引き分け
NUM_LABELS = 3
TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-small"


@dataclass(slots=True)
class TrainConfig:
    """v1 の学習・推論で使う設定。"""

    # --- モデルと入力 ---------------------------------------------------
    model_name: str = DEFAULT_MODEL_NAME
    num_labels: int = NUM_LABELS
    # DeBERTa-v3 は相対位置埋め込みが 512 までなので、ここは 512 を超えないこと。
    max_length: int = 512
    # 入力長が足りないときにどこを切り捨てるかの配分。
    # prompt_ratio: プロンプトに割り当てるトークン予算の割合（残りを A/B で折半する）
    prompt_ratio: float = 0.15
    # head_ratio: 応答を切り詰めるとき、先頭側をどれだけ残すか（末尾は結論部分なので残す）
    head_ratio: float = 0.7
    # 応答 A/B を入れ替えてラベルも反転させる拡張の確率（位置バイアスを消す）
    swap_augment_p: float = 0.5

    # --- 最適化 ---------------------------------------------------------
    batch_size: int = 8
    eval_batch_size: int = 16
    epochs: int = 3
    # 0 未満なら無効。総ステップ数を絞って短い実験できるようにする
    max_steps: int = -1
    # 1.0 未満なら学習バッチを絞る（1 エポックを短く回して結果を見るため）
    limit_train_batches: float = 1.0
    accumulate_grad_batches: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    # ラベルスムージングは log loss（提出スコア）の校正に効く
    label_smoothing: float = 0.05
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    # base 以降へ大きくしたときに VRAM を節約する（学習が遅くなる）
    gradient_checkpointing: bool = False

    # --- 推論（predict.py 用） -----------------------------------------
    temperature: float = 1.0
    swap_tta: bool = True

    # --- 実行系 ---------------------------------------------------------
    version: str = "v1"
    accelerator: str = "auto"
    # "auto" にすると利用可能な GPU を自動選択する。1枚に絞りたい場合は --devices 2 のように指定する
    devices: str = "auto"
    precision: str = "bf16-mixed"
    num_workers: int = 4
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def add_arguments(
        cls, parser: argparse.ArgumentParser, skip: set[str] | None = None
    ) -> None:
        """dataclass の各フィールドを argparse の引数として登録する。"""
        skip = set(skip or ()) | {
            "num_labels"
        }  # num_labels は提出形式に直結するため変更不可
        hints = get_type_hints(cls)
        for f in fields(cls):
            if f.name in skip:
                continue
            flag = f"--{f.name.replace('_', '-')}"
            annotation = hints[f.name]
            if annotation is bool:
                parser.add_argument(
                    flag,
                    dest=f.name,
                    action=argparse.BooleanOptionalAction,
                    default=f.default,
                    help=f"{f.name} (default: {f.default})",
                )
            else:
                parser.add_argument(
                    flag,
                    dest=f.name,
                    type=annotation,
                    default=f.default,
                    help=f"{f.name} (default: {f.default})",
                )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainConfig:
        values = {}
        for f in fields(cls):
            if f.name == "num_labels":
                continue
            values[f.name] = getattr(args, f.name, f.default)
        return cls(**values)


def build_arg_parser(
    prog: str, skip: set[str] | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    TrainConfig.add_arguments(parser, skip=skip)
    return parser
