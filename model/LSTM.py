from torch import nn, zeros, optim, device
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self, hidden_layer_size=100, input_size=3, d=device('cuda'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.loss_function = nn.MSELoss()
        self.evaluation_data = []
        self.d = d

    def init_hidden(self):
        return zeros(1, 1, self.hidden_layer_size, device=self.d), \
               zeros(1, 1, self.hidden_layer_size, device=self.d)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
