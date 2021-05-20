import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import tensor
from pytorch_lightning.metrics.functional.classification import stat_scores
from pytorch_lightning import metrics

result_map = {
    "b6l10.0d9": "L = 6",
    "b8l10.0d9": "BN = 8",
    "b10l0.1d9": "b10l0.1d9",
    "b10l10.0d9": "L = 10",
    "b10l10.0d1": "L = 10",
    "b10l10.0d5": "b10l10.0d5",
    "b12l10.0d9": "BN = 12",
    "b14l10.0d9": "λ = 10",
    "b14l1.0d9": "λ = 1",
    "b14l100.0d9": "λ = 100",
    "b16l10.0d9": "BN = 16",
    "b18l10.0d9": "L = 18",
    "gak": "GAK",
    "iForest": "iForest",
    "DTW": "DTW",
    "b20l10.0d9": "L = 20",
    "b4l10.0d9": "λ = 10",
    "b14l0.1d9": "λ = 0.1",
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "test": "test",
    "test2": "test2",
    "b20l10.0d1": "b20l10.0d1",
    "b20l10.0d3": "b20l10.0d3",
}
#
# graph = ["b6l10.0d9", "b8l10.0d9", "b10l0.1d9", "b10l10.0d9", "b10l10.0d1", "b10l10.0d5", "b12l10.0d9",
#          "b14l10.0d9", "b14l100.0d9", "b16l10.0d9", "b18l10.0d9", "b20l10.0d9", "b4l10.0d9", "b14l0.1d9"]


graph = ["b20l10.0d1", ]
# graph = ["b10l10.0d9", "gak", "DTW", "iForest"]
# graph = ["b6l10.0d9", "b10l10.0d9", "b14l10.0d9", "b18l10.0d9", "b20l10.0d9"]
# graph = ["DTW"]

result = []
accuracy = metrics.Accuracy()
F1 = metrics.F1(num_classes=1)
precision = metrics.Precision(num_classes=1)
recall = metrics.Recall(num_classes=1)
all_data = []
all_data2 = []
b = []
for g in graph:
    data = []
    all = []
    f = open(os.path.realpath('..') + '/result/' + g, 'r')
    all += f.readlines()

    for l in all:
        data.append((float(l.split(',')[0]), float(l.split(',')[1])))
    error = 5
    tpr = []
    fpr = []
    recs = []
    perss = []
    best = (0, {})
    while error >= -5:
        pred = []
        y = []
        for record in data:
            d, e = record
            try:
                if g != "DTW":
                    pred.append(1 if d > error else 0)
                else:
                    pred.append(1 if 4 - d > error else 0)
            except:
                print(d)
            y.append(1 if e else 0)
        pred = tensor(pred)
        y = tensor(y)

        tps, fps, tns, fns, sups = stat_scores(pred, y, class_index=1)
        tps = tps.item()
        fps = fps.item()
        tns = tns.item()
        fns = fns.item()
        rec = recall(pred, y)
        pres = precision(pred, y)
        f1 = F1(pred, y)
        acc = accuracy(pred, y)
        tprate = tps / (tps + fns)
        fprate = fps / (fps + tns)
        # print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
        #       fps.item() / (fps.item() + tns.item()))
        # print(
        #     f'Error: {error} tp:{tps} fp:{fps} tn:{tns} fn:{fns}\n'
        #     + f'TP Rate: {tprate * 100}% FP Rate: {fprate * 100}%'
        #     + f'Accuracy: {acc * 100}% Precision:{pres * 100}% Recall:{rec * 100}% F1:{f1 * 100}%'
        #       f'\n-------------------------------------------------------------------------')
        if f1 > best[0]:
            best = (f1,
                    {"f1": f1, "rec": rec, "pres": pres, "tprate": tprate, "fprate": fprate, "tps": tps, "fps": fps,
                     "tns": tns,
                     "fns": fns, "error": error, "acc": acc
                     })

        tpr.append(tprate)
        fpr.append(fprate)

        if tps == 0 and fps == 0:
            pass
        elif rec.item() != 0 or pres.item() != 0:
            recs.append(rec.item())
            perss.append(pres.item())
        # error -= 100
        if error > 99:
            error = 4.9
        elif error > 5:
            error -= 1
        elif error > 2:
            error -= 0.1
        else:
            error -= 0.01
    print(f'{g}:{best}')
    b.append(best[1]['f1'].item())
    all_data.append((tpr, fpr))
    all_data2.append((perss, recs))
# Change the style of plot
plt.style.use('seaborn-dark')

# Create a color palette
palette = ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']
linestyle = ['solid', (0, (3, 1, 1, 1)), '--', '-.', ':']
num = 0
plt.figure(figsize=(10, 5), dpi=450)
for num in range(len(all_data)):
    d = all_data[num]
    d2 = all_data2[num]
    # plt.subplot(121, aspect='0.8', autoscale_on=True)
    # plt.plot(d[1], d[0], linestyle=linestyle[num], color=palette[num], linewidth=2, alpha=1,
    #          label=f'{result_map[graph[num]]} (F1:{round(b[num], 3)})')
    # # plt.title("ROC Curve", loc='left', fontsize=14, fontweight=1, color='black')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.subplot(121, aspect='0.8', autoscale_on=True)
    plt.plot(d[1], d[0], linestyle=linestyle[num], color=palette[num], linewidth=2, alpha=1,
             label=f'{result_map[graph[num]]} (F1:{round(b[num], 3)})')
    # plt.title("Precision-Recall Curve", loc='left', fontsize=14, fontweight=1, color='black')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
plt.show()
