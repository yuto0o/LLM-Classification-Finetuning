"""生データの読み込み・ラベル化・層別分割."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kaggle.models.v1.config import TARGET_COLUMNS
from kaggle.models.v1.utils.type import ChatData
from kaggle.utils.constants import DataPaths, ModelPaths

INPUT_COLUMNS = ["id", "prompt", "response_a", "response_b"]


def load_chat_frame(path: Path, *, with_label: bool) -> pd.DataFrame:
    """CSV を読み込み、`label` 列（0: A / 1: B / 2: tie）を付与した DataFrame を返す.

    test.csv には target カラムがないので `with_label=False` で読む。
    """
    df = pd.read_csv(path)

    missing = [column for column in INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} に必須列がありません: {missing}")

    if not with_label:
        return df

    missing_targets = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing_targets:
        raise ValueError(f"{path} に target 列がありません: {missing_targets}")

    # 元の実装は argmax だけで、フラグが 0 個 / 2 個以上の行を黙って 0 (Model A) に
    # 変換していた。ここでは明示的に検証する
    one_hot = df[TARGET_COLUMNS].to_numpy(dtype=bool)
    invalid = df.index[one_hot.sum(axis=1) != 1]
    if len(invalid) > 0:
        raise ValueError(
            f"{path}: target が 1 個だけ真ではない行が {len(invalid)} 行あります "
            f"(例: id={df.loc[invalid[0], 'id']})"
        )
    df["label"] = one_hot.argmax(axis=1)
    return df


def validate_frame(df: pd.DataFrame, *, sample_size: int = 1000, seed: int = 0) -> None:
    """欠損がないかを全行に対して検査し、抜粋を ChatData で検証する."""
    columns = INPUT_COLUMNS + (["label"] if "label" in df.columns else [])
    for column in columns:
        if df[column].isna().any():
            raise ValueError(
                f"列 {column} に欠損があります: {int(df[column].isna().sum())} 件"
            )

    records = df.to_dict(orient="records")
    if len(records) > sample_size:
        records = random.Random(seed).sample(records, sample_size)
    for record in records:
        ChatData.model_validate(record)


def split_data(
    path: Path = DataPaths.RAW_DATA_DIR / "train.csv",
    version: str = "v1",
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Splits the data into stratified folds for cross-validation.

    Args:
        path (Path): path to the raw csv file
        version (str): 分割結果を保存する models/<version>/data の version
        n_splits (int, optional): _description_. Defaults to 5.
        random_state (int, optional): _description_. Defaults to 42.
    """

    df = load_chat_frame(path, with_label=True)
    validate_frame(df)

    # split the data into stratified folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # use fold_idx = 0 for validation, fold_idx = 1 for testing, and the rest for training
    val_fold = 0
    test_fold = 1

    train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)].reset_index(
        drop=True
    )
    val_df = df[df["fold"] == val_fold].reset_index(drop=True)
    test_df = df[df["fold"] == test_fold].reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print(
        "\nTrain Label Distribution:\n", train_df["label"].value_counts(normalize=True)
    )
    print("Val Label Distribution:\n", val_df["label"].value_counts(normalize=True))
    print("Test Label Distribution:\n", test_df["label"].value_counts(normalize=True))

    # save the split data to the models/v1/data directory
    path_to_save = ModelPaths.BASE_DIR / version / "data"
    path_to_save.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(path_to_save / "train_split.csv", index=False)
    val_df.to_csv(path_to_save / "val_split.csv", index=False)
    test_df.to_csv(path_to_save / "test_split.csv", index=False)

    print(f"Data saved to {path_to_save}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="data_split")
    parser.add_argument(
        "--input",
        type=Path,
        default=DataPaths.RAW_DATA_DIR / "train.csv",
        help="train.csv のパス",
    )
    parser.add_argument(
        "--version", default="v1", help="分割結果の保存先 models/<version>/data"
    )
    parser.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    parser.add_argument("--seed", dest="random_state", type=int, default=42)
    args = parser.parse_args(argv)

    split_data(
        path=args.input,
        version=args.version,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
