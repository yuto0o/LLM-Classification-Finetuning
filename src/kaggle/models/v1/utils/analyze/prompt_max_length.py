import pandas as pd

from kaggle.utils.constants import DataPaths

dataset = pd.read_csv(DataPaths.RAW_DATA_DIR / "train.csv")
print("Max length of response_a:", dataset["response_a"].str.len().max())
print("Max length of response_b:", dataset["response_b"].str.len().max())
print("Max length of prompt:", dataset["prompt"].str.len().max())
