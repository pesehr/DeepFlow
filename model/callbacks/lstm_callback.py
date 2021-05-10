import os

from pytorch_lightning.callbacks import Callback


# from evaluation.validator import Validator


class LSTMCallback(Callback):
    def __init__(self, name):
        self.name = name

    def on_test_end(self, trainer, pl_module):
        f = open(os.path.realpath('') + f'/result/{self.name}', 'w')
        for i in range(0, len(pl_module.evaluation_data)):
            ev = pl_module.evaluation_data[i]
            f.write(str(ev[0]) + ',' + str(ev[1]) + '\n')
