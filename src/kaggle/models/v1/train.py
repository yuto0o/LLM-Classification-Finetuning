import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# パス等のインポートは環境に合わせて調整してください
from transformers import AutoTokenizer

from kaggle.models.v1.generator.deberta_generator import ChatLightningModule
from kaggle.models.v1.utils.type import ChatDataset
from kaggle.utils.constants import ModelPaths

logger = TensorBoardLogger(ModelPaths.BASE_DIR / "v1" / "tb_logs", name="my_model")


def train():

    MODEL_NAME = "microsoft/deberta-v3-small"
    BATCH_SIZE = 8
    MAX_LENGTH = 512
    # load data
    data_path = ModelPaths.BASE_DIR / "v1" / "data"
    train_df = pd.read_csv(data_path / "train_split.csv")
    val_df = pd.read_csv(data_path / "val_split.csv")

    test_df = pd.read_csv(data_path / "test_split.csv")  # Assuming you have a test set

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Dataset & DataLoader
    train_dataset = ChatDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = ChatDataset(val_df, tokenizer, MAX_LENGTH)

    test_dataset = ChatDataset(test_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = ChatLightningModule(model_name=MODEL_NAME, lr=1e-5)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=2)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=[2],
        precision="32-true",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)


if __name__ == "__main__":
    train()
