import lightning.pytorch as pl
from torch import optim
from transformers import AutoModelForSequenceClassification


class ChatLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int = 3, lr: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.float()
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer
