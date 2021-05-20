from torch import nn, zeros, optim, device
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
