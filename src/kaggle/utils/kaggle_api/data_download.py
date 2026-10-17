import kagglehub

from kaggle.utils.constants import DataPaths

DataPaths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version
# kagglehub 1.x では出力先ディレクトリは output_dir で指定する。
# （旧 API の path はコンペ内の「ファイル相対パス」を指すため、そのまま渡ると失敗する）
path = kagglehub.competition_download(
    "llm-classification-finetuning",
    output_dir=str(DataPaths.RAW_DATA_DIR),
    force_download=True,
)

print("Path to competition files:", path)
for name in ("train.csv", "test.csv", "sample_submission.csv"):
    file_path = DataPaths.RAW_DATA_DIR / name
    print(f"  {name}: {'OK' if file_path.exists() else '見つかりません'}")
