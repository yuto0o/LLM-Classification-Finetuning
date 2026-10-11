from pathlib import Path


class DataPaths:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE_DIR / "utils" / "raw_data"


class ModelPaths:
    BASE_DIR = Path(__file__).resolve().parent.parent / "models"
