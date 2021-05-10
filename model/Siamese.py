import numpy as np
from torch import nn, zeros, device, cat, tensor, std
from tslearn.metrics import gak
from dtaidistance import dtw
from model.trash.LSTM import LSTM
import torch
from sklearn.ensemble import IsolationForest


class Siamese(LSTM):
    def __init__(self, hidden_layer_size=100, battle_neck=20, feature_len=1, observe_len=5, label_len=1,
                 objects_len=5,
                 lamda=10,
                 d=device('cuda'), *args,
                 **kwargs):
        super().__init__(hidden_layer_size, device, *args, **kwargs)
        self.LSTM1 = nn.LSTM(feature_len, hidden_layer_size)
        self.lamda = lamda
        self.encoder1 = nn.Linear(hidden_layer_size, battle_neck)
        self.LSTM2 = nn.LSTM(battle_neck, hidden_layer_size)
        self.encoder2 = nn.Linear(hidden_layer_size, feature_len)
        self.battle_neck = battle_neck
        self.feature_len = feature_len
        self.hidden_layer_size = hidden_layer_size
        self.d = d
        # self.observe_len = observe_len
        self.label_len = label_len
        self.objects_len = objects_len
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-2)

    def forward(self, batch):
        # LSTM 1
        h1, h2 = self.init_hidden()
        observe_len = len(batch[0])
        encoder_outputs, (_, _) = self.LSTM1(batch.view(observe_len, 1, self.feature_len),
                                             (h1, h2))

        # Encoder 1
        encoded = self.encoder1(encoder_outputs[-1])
        encoded = encoded.view(1, 1, self.battle_neck)
        # # LSTM 2
        h1, h2 = self.init_hidden()
        output = []
        for i in range(0, observe_len):
            LSTM_output, (h1, h2) = self.LSTM2(encoded, (h1, h2))
            LSTM_output = LSTM_output[-1].view(1, 1, self.hidden_layer_size)
            # Encode 2
            decoded = self.encoder2(LSTM_output)
            output.append(decoded)
        return output, encoded

    def get_loss(self, x, y):
        return self.loss_function(x, y)

    def validation(self, batch):
        loss = 0
        decoded = []
        for j in range(0, self.objects_len):
            output, d = self(batch[0][0][j])
            decoded.append(d[0][0])
            output = cat(output).view(self.feature_len * len(batch[0][0][j][0]))
            list = batch[0][0][j][0].view(self.feature_len * len(batch[0][0][j][0]))
            loss += self.loss_function(output, list)
        loss2 = 0
        for i in range(0, self.objects_len):
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i], decoded[j])
                    loss2 += (-l + 1)
        loss = loss + (loss2 / self.lamda)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.validation(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.validation(batch)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # a = [0, 0, 0, 0, 0, 0]
        # d = [0, 0, 0, 0, 0, 0]
        # for j in range(0, self.objects_len):
        #     a[j] = np.array([(batch[0][0][j][0][k].item()-300)/500 for k in range(len(batch[0][0][j][0]))])
        # for j in range(0, self.objects_len):
        #     for i in range(0, self.objects_len):
        #         if i != j:
        #             d[j] += gak(a[i], a[j])
        # for i in range(0, self.objects_len):
        #     self.evaluation_data.append((4 - d[i], 1 if batch[1].item() == i else 0))

        # a = [0, 0, 0, 0, 0, 0]
        min = 1000
        for j in range(0, self.objects_len):
            #     a[j] = np.array([(batch[0][0][j][0][k].item() - 300) / 500 for k in range(len(batch[0][0][j][0]))])
            if len(batch[0][0][j][0]) < min:
                min = len(batch[0][0][j][0])
        # i = IsolationForest(random_state=0).fit([a[0][:min], a[1][:min], a[2][:min], a[3][:min], a[4][:min]])
        # i = i.score_samples([a[0][:min], a[1][:min], a[2][:min], a[3][:min], a[4][:min]])
        # for j in range(0, self.objects_len):
        #     self.evaluation_data.append((-i[j], 1 if batch[1].item() == j else 0))

        decoded = []
        for j in range(0, self.objects_len):
            o, d = self(batch[0][0][j])
            decoded.append(d)
        o = []
        for i in range(0, self.objects_len):
            loss = 0
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i][0][0], decoded[j][0][0])
                    # l = self.similarity(batch[0][0][j][0][:min], batch[0][0][i][0][:min])
                    loss += l
            o.append(loss)
        # ma = max(o)
        # mi = min(o)
        std = torch.std(tensor([o[j].item() for j in range(0, len(o))]))
        mean = torch.mean(tensor([o[j].item() for j in range(0, len(o))]))
        o2 = []
        # mo = 0
        for i in range(0, self.objects_len):
            # mo += o[i].item()
            o2.append(o[i])
            # o[i] = ((o[i] - std) / mean).item()
            o[i] = o[i].item()
        # mo /= self.objects_len
        for i in range(0, self.objects_len):
            self.evaluation_data.append(((4 - o[i]) * 100, 1 if batch[1][j].item() == i else 0))
            # self.evaluation_data.append((loss.item(), 1 if batch[2].item() == i else 0))
