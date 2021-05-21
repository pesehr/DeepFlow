from torch import nn

from model.LSTM import LSTM


class Decoder(nn.Module):
    def __init__(self, input_dim, n_features):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, self.n_features)
        x = x.reshape((self.n_features, seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((seq_len, self.hidden_dim))
        return self.output_layer(x)
