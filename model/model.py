import torch
import torchmetrics
import transformers
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # self.log("train_loss", loss)
        metrics = {"loss": loss, "train_pearson": torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # self.log("val_loss", loss)
        # self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        metrics = {"val_loss": loss, "val_pearson": torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer