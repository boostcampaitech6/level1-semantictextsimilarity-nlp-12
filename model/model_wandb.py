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

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        # Freeze all layers
        # for param in self.plm.parameters():
            # param.requires_grad = False

        self.loss_func = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['target']
        logits = self(x, attention_mask)
        loss = self.loss_func(logits, y.float())

        self.log('train_loss', loss)
        self.log('train_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['target']
        logits = self(x, attention_mask)
        loss = self.loss_func(logits, y.float())

        self.log('val_loss', loss)
        self.log('val_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['target']
        logits = self(x, attention_mask)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch['input_ids']
        attention_mask = batch['attention_mask']
        logits = self(x, attention_mask)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        # return [optimizer], [scheduler]
