"""DeBERTa-v3 を使う Lightning モジュール.

大会指標が multi-class log loss なので、監視する `val_loss` は温度・ラベル
スムージングを適用していない素の交差エントロピーにしている（提出スコアと同じ値）。
最適化に使う損失は `*_loss_opt` として別にログに出す。
"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from kaggle.models.v1.config import TrainConfig


class ChatLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg: TrainConfig | dict,
        num_training_steps: int | None = None,
    ):
        super().__init__()
        # checkpoint 復元時は hparams の dict がそのまま渡される
        if isinstance(cfg, dict):
            cfg = TrainConfig(**cfg)
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg.to_dict()})

        # 重みは fp32 で読み込む（このチェックポイントは fp16 で読まれることがあり、
        # そのまま学習すると DeBERTa の attention が溢れて loss が nan になる）。
        # 混合精度は Trainer の precision に任せる。
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=cfg.num_labels, dtype=torch.float32
        )
        # from_pretrained は eval モードで返す。放置すると dropout が効かないまま学習する
        self.model.train()
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        # ラベルスムージングを効かせたい損失と、評価として見る素の CE を分ける
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # 段階名をそのままキーにすると nn.Module の `train` と衝突するので接頭辞を付ける
        self.metrics = nn.ModuleDict(
            {
                f"stage_{stage}": nn.ModuleDict(
                    {
                        "acc": MulticlassAccuracy(
                            num_classes=cfg.num_labels, average="micro"
                        ),
                        "f1": MulticlassF1Score(
                            num_classes=cfg.num_labels, average="macro"
                        ),
                    }
                )
                for stage in ("train", "val", "test")
            }
        )

        self.num_training_steps = num_training_steps
        self.lr = cfg.lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """提出用の確率。温度でなめらかにしてから softmax する（temperature=1 なら等価）."""
        return torch.softmax(logits.float() / self.cfg.temperature, dim=-1)

    def _shared_step(self, batch, stage: str):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
        )
        logits = outputs.logits.float()
        labels = batch["labels"]

        loss_opt = self.loss_fn(logits, labels)
        # 大会指標と同じ値。ModelCheckpoint / EarlyStopping はこれを見る
        logloss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        metrics = self.metrics[f"stage_{stage}"]
        acc = metrics["acc"](preds, labels)
        f1 = metrics["f1"](preds, labels)

        on_step = stage == "train"
        self.log(
            f"{stage}_loss",
            logloss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_loss_opt",
            loss_opt,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_acc",
            acc,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(f"{stage}_f1", f1, on_epoch=True, sync_dist=True)

        return loss_opt

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """温度適用後の確率を返す（提出ファイル生成用）."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
        )
        return self.probabilities(outputs.logits)

    def on_train_epoch_start(self) -> None:
        """epoch ごとに swap 拡張の当たり方を変える."""
        loaders = (
            self.trainer.datamodule.train_dataloader()
            if self.trainer.datamodule
            else None
        )
        if loaders is None:
            loaders = self.trainer.train_dataloader
        datasets = loaders if isinstance(loaders, list) else [loaders]
        for loader in datasets:
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        # bias と LayerNorm の重みは weight decay しない（BERT 系の定番の分け方）
        decay, no_decay = [], []
        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.dim() <= 1 else decay).append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        num_training_steps = self.num_training_steps
        if num_training_steps is None or num_training_steps <= 0:
            return optimizer

        num_warmup_steps = int(num_training_steps * self.cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
