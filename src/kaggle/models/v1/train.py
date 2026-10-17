"""v1 の学習を実行するスクリプト.

例::

    uv run python src/kaggle/models/v1/train.py                       # 既定（deberta-v3-small / 3 epochs）
    uv run python src/kaggle/models/v1/train.py --devices 2          # GPU を 1 枚に絞る
    uv run python src/kaggle/models/v1/train.py --max-steps 200     # 短い実験
"""

import json
import math
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from kaggle.models.v1.config import TrainConfig, build_arg_parser
from kaggle.models.v1.generator.deberta_generator import ChatLightningModule
from kaggle.models.v1.utils.data_split import load_chat_frame
from kaggle.models.v1.utils.type import ChatDataset, make_collate_fn
from kaggle.utils.constants import ModelPaths


def parse_devices(value: str):
    """`--devices auto` / `2` / `0,1` を Trainer に渡せる形にする."""
    if value.strip().lower() == "auto":
        return "auto"
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if all(part.isdigit() for part in parts):
        return [int(part) for part in parts] if len(parts) > 1 else int(parts[0])
    return value


def training_steps(cfg: TrainConfig, num_batches: int) -> int:
    """warmup / 学習率減衰のスケジュールを決めるための総ステップ数."""
    batches_per_epoch = num_batches
    if cfg.limit_train_batches < 1.0:
        batches_per_epoch = max(1, int(num_batches * cfg.limit_train_batches))
    steps_per_epoch = math.ceil(batches_per_epoch / cfg.accumulate_grad_batches)
    if cfg.max_steps > 0:
        return cfg.max_steps
    return steps_per_epoch * cfg.epochs


def train(cfg: TrainConfig) -> Path:
    data_dir = ModelPaths.BASE_DIR / cfg.version / "data"
    checkpoint_dir = ModelPaths.BASE_DIR / cfg.version / "checkpoints"
    log_dir = ModelPaths.BASE_DIR / cfg.version / "tb_logs"
    for directory in (data_dir, checkpoint_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(cfg.seed, workers=True)

    # 分割は data_split.py が作ったものを使う（fold 0 = val, fold 1 = test）
    train_df = load_chat_frame(data_dir / "train_split.csv", with_label=True)
    val_df = load_chat_frame(data_dir / "val_split.csv", with_label=True)
    test_df = load_chat_frame(data_dir / "test_split.csv", with_label=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collate_fn = make_collate_fn(tokenizer)

    # 学習時だけ swap 拡張を効かせる（検証・test は生の順序で評価する）
    train_dataset = ChatDataset(
        train_df,
        tokenizer,
        max_length=cfg.max_length,
        prompt_ratio=cfg.prompt_ratio,
        head_ratio=cfg.head_ratio,
        swap_augment_p=cfg.swap_augment_p,
        seed=cfg.seed,
    )
    val_dataset = ChatDataset(
        val_df,
        tokenizer,
        max_length=cfg.max_length,
        prompt_ratio=cfg.prompt_ratio,
        head_ratio=cfg.head_ratio,
    )
    test_dataset = ChatDataset(
        test_df,
        tokenizer,
        max_length=cfg.max_length,
        prompt_ratio=cfg.prompt_ratio,
        head_ratio=cfg.head_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = ChatLightningModule(
        cfg, num_training_steps=training_steps(cfg, len(train_loader))
    )

    # 大会指標が log loss なので、最良チェックポイントと早期打ち切りは val_loss（=素の CE）で決める
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best-{cfg.version}-epoch{{epoch:02d}}-valloss{{val_loss:.4f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=2, verbose=True
    )

    logger = TensorBoardLogger(log_dir, name=cfg.model_name.split("/")[-1])

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        limit_train_batches=cfg.limit_train_batches,
        accelerator=cfg.accelerator,
        devices=parse_devices(cfg.devices),
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
    )

    # 何を実験したかを残しておく（後から比較できるようにする）
    (checkpoint_dir / "train_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss (log loss): {checkpoint_callback.best_model_score}")
    return Path(checkpoint_callback.best_model_path)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser("train")
    args = parser.parse_args(argv)
    train(TrainConfig.from_args(args))


if __name__ == "__main__":
    main()
