import kagglehub

from src.kaggle.utils.constants import DataPaths

DataPaths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version
path = kagglehub.competition_download(
    "llm-classification-finetuning",
    path=str(DataPaths.RAW_DATA_DIR),
    force_download=True,
)

print("Path to competition files:", path)
