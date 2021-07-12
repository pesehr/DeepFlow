from torch import nn, tanh

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.LSTM import LSTM


class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features=1, latent_size=40):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, latent_size)
        self.decoder = Decoder(latent_size, n_features)
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        seq_len = len(x[0])
        latent = self.encoder(x, seq_len)
        reconstructed = self.decoder(latent, seq_len)
        return latent, tanh(self.loss(x[0], reconstructed))
