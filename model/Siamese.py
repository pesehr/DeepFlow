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
    def __init__(self, latent_size=40, feature_len=1, objects_len=5, lamda=10, d=device('cuda'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autoencoder = RecurrentAutoencoder(n_features=feature_len)

        self.lamda = lamda
        self.latent_size = latent_size
        self.feature_len = feature_len
        self.d = d
        self.objects_len = objects_len

        self.similarity = nn.MSELoss(reduction='mean')

    def forward(self, batch):
        decoded, error = self.autoencoder(batch)
        return decoded, error

    def get_loss(self, x, y):
        return self.loss_function(x, y)

    def validation(self, batch):
        reconstruction_loss = 0
        decoded = []
        for j in range(0, self.objects_len):
            d, error = self(batch[0][0][j])
            decoded.append(d[0])
            reconstruction_loss += error
        similarity_loss = 0
        for i in range(0, self.objects_len):
            for j in range(i + 1, self.objects_len):
                l = self.similarity(decoded[i], decoded[j])
                similarity_loss += l
        reconstruction_loss /= self.objects_len
        similarity_loss /= (self.objects_len * (self.objects_len - 1)) / 2
        loss = similarity_loss + reconstruction_loss * self.lamda
        return loss, reconstruction_loss, similarity_loss

    def training_step(self, batch, batch_idx):
        loss, loss1, loss2 = self.validation(batch)
        self.log('total loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('reconstruct loss', loss1, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('similarity loss', loss2, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss1, loss2 = self.validation(batch)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # GAK/DTW
        # a = [0, 0, 0, 0, 0, 0]
        # d = [0, 0, 0, 0, 0, 0]
        # for j in range(0, self.objects_len):
        #     a[j] = np.array([batch[0][0][j][0][k].item() for k in range(len(batch[0][0][j][0]))])
        # for j in range(0, self.objects_len):
        #     for i in range(j, self.objects_len):
        #         d[j] += gak(a[i], a[j])
        # # for i in range(0, self.objects_len):
        # self.evaluation_data.append((sum(d), 1 if sum(batch[1]).item() > -5 else 0))

        # IForest
        # a = [0, 0, 0, 0, 0, 0]
        # mi = 1000
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
            # if len(batch[0][0][j][0]) < mi:
            #     mi = len(batch[0][0][j][0])
            l1.append(e)
            decoded.append(d)
        o = []
        for i in range(0, self.objects_len):
            loss = 0
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i][0], decoded[j][0])
                    loss += l
            o.append(loss)

        self.evaluation_data.append((20 - sum(o).item(), 1 if sum(batch[1]).item() > -5 else 0))
