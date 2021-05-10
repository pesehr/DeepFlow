import os

from torch import tensor
from pytorch_lightning.metrics.functional.classification import stat_scores
import matplotlib.pyplot as plt
from pytorch_lightning import metrics


class Validator:
    def __init__(self, data):

        self.accuracy = metrics.Accuracy()
        self.F1 = metrics.F1(num_classes=1)
        self.precision = metrics.Precision(num_classes=1)
        self.recall = metrics.Recall(num_classes=1)
        self.test(data)

    # self.Precision = Precision(num_classes=2).to(device("cuda", 0))
    # self.Recall = Recall(num_classes=2).to(device("cuda", 0))
    # self.ROC = classification.ROC(pos_label=1)
    # self.ROC = classification.ROC(pos_label=1)

    def test(self, data):
        speed_err = 1
        error = 4
        tpr = []
        fpr = []
        best = (0, {})
        while error >= -2:
            pred = []
            y = []
            for record in data:
                d, e = record
                pred.append(1 if d > error else 0)
                y.append(1 if e else 0)
            error -= 0.001

            pred = tensor(pred)
            y = tensor(y)

            tps, fps, tns, fns, sups = stat_scores(pred, y, class_index=1)
            tps = tps.item()
            fps = fps.item()
            tns = tns.item()
            fns = fns.item()
            rec = self.recall(pred, y)
            pres = self.precision(pred, y)
            f1 = self.F1(pred, y)
            acc = self.accuracy(pred, y)
            tprate = tps / (tps + fns)
            fprate = fps / (fps + tns)
            # print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
            #       fps.item() / (fps.item() + tns.item()))
            print(
                f'Error: {error} tp:{tps} fp:{fps} tn:{tns} fn:{fns}\n'
                + f'TP Rate: {tprate * 100}% FP Rate: {fprate * 100}%'
                + f'Accuracy: {acc * 100}% Precision:{pres * 100}% Recall:{rec * 100}% F1:{f1 * 100}%'
                  f'\n-------------------------------------------------------------------------')
            if f1 > best[0]:
                best = (f1,
                        {"f1": f1, "rec": rec, "pres": pres, "tprate": tprate, "fprate": fprate, "tps": tps, "fps": fps,
                         "tns": tns,
                         "fns": fns, "error": error, "acc": acc
                         })

            tpr.append(tprate)
            fpr.append(fprate)
        plt.plot(fpr, tpr)

        plt.grid(True)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LSTMEncoderLSTM')
        plt.text(0.8, 0, f'AUC:{metrics.functional.classification.auc(tensor(fpr), tensor(tpr)):.2f}', bbox=dict(
            boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(best[1]["fprate"], best[1]["tprate"] - 0.42,
                 f'Threshold:{best[1]["error"]}\ntp:{best[1]["tps"]} fp:{best[1]["fps"]} tn:{best[1]["tns"]} fn:{best[1]["fns"]}\n'
                 + f'TP Rate: {best[1]["tprate"] * 100:.2f}%\nFP Rate: {best[1]["fprate"] * 100:.2f}%\n'
                 + f'Accuracy: {best[1]["acc"] * 100:.2f}%\nPrecision:{best[1]["pres"] * 100:.2f}%\nRecall:{best[1]["rec"] * 100:.2f}%\n'
                 + f'F1:{best[1]["f1"] * 100:.2f}%',
                 bbox=dict(
                     boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=14)

        plt.show()


data = []
f = open(os.path.realpath('..') + '/evaluation.log', 'r')
for l in f.readlines():
    data.append((float(l.split(',')[0]), float(l.split(',')[1])))
Validator(data)
