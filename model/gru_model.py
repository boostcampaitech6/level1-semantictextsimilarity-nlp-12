import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

class GRUModel(pl.LightningModule):
    #GRU를 얹기 위해서 hidden_size, dropout_prob가 필요한데 hidden_size는 huggingface에서 가져온 모델의 히든스테이트 크기에 달림.
    def __init__(self, model_name, lr, dropout_prob=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)#, output_hidden_states=True)
        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()
        # Bidirectional GRU, dropout, linear, tanh를 얹을 예정이다.
        # BERT계열 모델의 hidden_size는 보통 768이다.
        hidden_size = self.plm.config.hidden_size
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size//2, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                    out_features=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        output = self.plm(x)
        output = output.last_hidden_state
        _, last_gru_state = self.gru(output)
        last_hidden_state = torch.cat((last_gru_state[-1], last_gru_state[-2]), dim=-1)
        logits = self.tanh(last_hidden_state)
        logits = self.dropout(logits)
        logits = self.linear(logits)
        return logits
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # self.log("train_loss", loss); return loss;에서 수정됨.
        metrics = {"loss": loss, "train_pearson": torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

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