from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kaggle.utils.constants import DataPaths, ModelPaths


def split_data(
    path: Path = DataPaths.RAW_DATA_DIR / "train.csv",
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Splits the data into stratified folds for cross-validation.

    Args:
        path (Path): path to the raw csv file
        n_splits (int, optional): _description_. Defaults to 5.
        random_state (int, optional): _description_. Defaults to 42.
    """

    df = pd.read_csv(path)
    # make label column
    # 0 for Model A, 1 for Model B, 2 for tied
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    df["label"] = np.argmax(df[target_cols].values, axis=1)

    # split the data into stratified folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
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

    # train_records = [
    #     ChatData.model_validate(row) for row in train_df.to_dict(orient="records")
    # ]
    # val_records = [
    #     ChatData.model_validate(row) for row in val_df.to_dict(orient="records")
    # ]
    # test_records = [
    #     ChatData.model_validate(row) for row in test_df.to_dict(orient="records")
    # ]

    print(
        "\nTrain Label Distribution:\n", train_df["label"].value_counts(normalize=True)
    )
    print("Val Label Distribution:\n", val_df["label"].value_counts(normalize=True))
    print("Test Label Distribution:\n", test_df["label"].value_counts(normalize=True))

    # save the split data to the models/v1/data directory
    path_to_save = ModelPaths.BASE_DIR / "v1" / "data"
    path_to_save.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(path_to_save / "train_split.csv", index=False)
    val_df.to_csv(path_to_save / "val_split.csv", index=False)
    test_df.to_csv(path_to_save / "test_split.csv", index=False)

    print(f"Data saved to {path_to_save}")


if __name__ == "__main__":
    split_data()
