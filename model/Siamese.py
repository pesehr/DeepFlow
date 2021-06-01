import numpy as np
# from sklearn.ensemble import IsolationForest
from torch import nn, zeros, device, cat, tensor, std
# from tslearn.metrics import gak
# from dtaidistance import dtw
import torch

# from sklearn.ensemble import IsolationForest

from model.LSTM import LSTM
from model.LSTMAutoencoder import RecurrentAutoencoder


class Siamese(LSTM):
    def __init__(self, hidden_layer_size=10, battle_neck=10, feature_len=2, observe_len=5, label_len=1,
                 objects_len=5,
                 lamda=10,
                 d=device('cuda'), *args,
                 fun=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.autoencoder = RecurrentAutoencoder(n_features=feature_len)

        # self.LSTM1 = nn.LSTM(feature_len, hidden_layer_size)
        self.lamda = 10e-5
        # self.encoder1 = nn.Linear(hidden_layer_size, battle_neck)
        # self.LSTM2 = nn.LSTM(battle_neck, feature_len)
        # self.encoder2 = nn.Linear(hidden_layer_size, feature_len)
        self.battle_neck = battle_neck
        self.feature_len = feature_len
        self.hidden_layer_size = hidden_layer_size
        self.d = d
        # self.observe_len = observe_len
        self.label_len = label_len
        self.objects_len = objects_len
        self.fun = fun

        # if self.fun == 1:
        #     self.similarity = nn.CosineSimilarity(dim=0, eps=1e-7)
        # else:
        self.similarity = nn.MSELoss(reduction='mean')

    def forward(self, batch):
        decoded, error = self.autoencoder(batch)
        return decoded, error

    def get_loss(self, x, y):
        return self.loss_function(x, y)

    def validation(self, batch):
        loss1 = 0
        decoded = []
        for j in range(0, self.objects_len):
            d, error = self(batch[0][0][j])
            decoded.append(d[0])
            loss1 += error
        loss2 = 0
        for i in range(0, self.objects_len):
            for j in range(i + 1, self.objects_len):
                l = self.similarity(decoded[i], decoded[j])
                loss2 += -l
        loss1 /= 5
        loss2 /= 10
        loss = loss1 + loss2
        return loss, loss1, loss2

    def training_step(self, batch, batch_idx):
        loss, loss1, loss2 = self.validation(batch)
        # self.log('total loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        # self.log('reconstruct loss', loss1, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        # self.log('similarity loss', loss2, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss1, loss2 = self.validation(batch)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('reconstruct loss', loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('similarity_loss', loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # a = [0, 0, 0, 0, 0, 0]
        # d = [0, 0, 0, 0, 0, 0]
        # for j in range(0, self.objects_len):
        #     a[j] = np.array([batch[0][0][j][0][k].item() for k in range(len(batch[0][0][j][0]))])
        # for j in range(0, self.objects_len):
        #     for i in range(j, self.objects_len):
        #         d[j] += gak(a[i], a[j])
        # # for i in range(0, self.objects_len):
        # self.evaluation_data.append((sum(d), 1 if sum(batch[1]).item() > -5 else 0))

        # a = [0, 0, 0, 0, 0, 0]
        mi = 1000
        # for j in range(0, self.objects_len):
        #     a[j] = np.array([batch[0][0][j][0][k].item() for k in range(len(batch[0][0][j][0]))])
        #     if len(batch[0][0][j][0]) < mi:
        #         mi = len(batch[0][0][j][0])
        # i = IsolationForest(random_state=0).fit([a[0][:mi], a[1][:mi], a[2][:mi], a[3][:mi], a[4][:mi]])
        # i = i.score_samples([a[0][:mi], a[1][:mi], a[2][:mi], a[3][:mi], a[4][:mi]])
        # # for j in range(0, self.objects_len):
        # self.evaluation_data.append((-sum(i), 1 if sum(batch[1]).item() > -5 else 0))

        decoded = []
        l1 = []
        for j in range(0, self.objects_len):
            d, e = self(batch[0][0][j])
            if len(batch[0][0][j][0]) < mi:
                mi = len(batch[0][0][j][0])
            l1.append(e)
            decoded.append(d)
        o = []
        for i in range(0, self.objects_len):
            loss = 0
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i][0], decoded[j][0])
                    # l = self.similarity(batch[0][0][j][0][:mi], batch[0][0][i][0][:mi])
                    loss += l
                    # loss += (l * -1) + 1
            o.append(loss)

        # # std = np.std(np.array(o))
        # # mean = np.mean(np.array(o))
        # std = torch.std(tensor([o[j].item() for j in range(0, len(o))]))
        # mean = torch.mean(tensor([o[j].item() for j in range(0, len(o))]))
        # # mino = min(o)
        # # maxo = max(o)
        for i in range(0, self.objects_len):
            # o[i] = ((o[i] - std) / mean).item()
            o[i] = o[i].item()
        # o[i] = ((o[i] - mino) / (maxo - mino)).item()

        # for i in range(0, self.objects_len):
        #     self.evaluation_data.append((4 - o[i], 1 if batch[1][i] == 1 else 0))
        if self.fun == 1:
            self.evaluation_data.append((20 - sum(o), 1 if sum(batch[1]).item() > -5 else 0))
        else:
            self.evaluation_data.append((sum(o), 1 if sum(batch[1]).item() > -5 else 0))
