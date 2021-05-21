from torch import nn

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.LSTM import LSTM


class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features=1, embedding_dim=40):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(embedding_dim, n_features)
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        seq_len = len(x[0])
        latent = self.encoder(x, seq_len)
        reconstructed = self.decoder(latent, seq_len)
        return latent, self.loss(x[0], reconstructed)
