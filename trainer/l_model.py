# from model import

import torch
import torch.optim as optim
import lightning as pl


class LightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss


def train(model, loss_fn, train_dataloader, val_dataloader, config):
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    pass
