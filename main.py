import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.Siamese import Siamese
from model.callbacks.lstm_callback import LSTMCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optparse
import re


def main(hparams):
    bottelNeck = int(hparams[0].hidden)
    lamda = float(hparams[0].lamda)
    ds = hparams[0].ds
    test = bool(hparams[0].test)
    print(bottelNeck, lamda, ds, test)
    datamodule = DrivingDataMadule(ds, 5800, 176, 10000)

    for filename in os.listdir("checkpoint"):
        if filename in ["b20l10d15"]:
            for c in os.listdir(f'checkpoint/{filename}/checkpoints'):
                print(f'{filename}/{c}')
                bottelNeck = int(re.search(r"b(.*?)l", filename).group(1))
                lamda = float(re.search(r"l(.*?)d", filename).group(1))
                ds = re.search(r"d(.*?).", filename)
                if test:
                    model = Siamese.load_from_checkpoint(
                        f'checkpoint/{filename}/checkpoints/{c}',
                        battle_neck=bottelNeck, lamda=lamda
                    )
                else:
                    model = Siamese(battle_neck=bottelNeck, lamda=lamda)

                checkpoint_callback = ModelCheckpoint(
                    monitor='validation_loss',
                    filename='LSTMEncoderLSTM--{v_num:02d}-{epoch:02d}-{validation_loss:.5f}-{train_loss:.5f}',
                )
                early_callback = EarlyStopping(monitor='validation_loss')

                trainer = pl.Trainer(gpus=1, max_epochs=100, accelerator='dp',
                                     callbacks=[LSTMCallback(name=f'b{bottelNeck}l{lamda}{ds.group()}'),
                                                checkpoint_callback],
                                     num_nodes=1)
                if test:
                    trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())
                else:
                    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-H', '--hidden',
                      action="store", dest="hidden",
                      help="Hidden State Size", default=10)
    parser.add_option('-l', '--lambda',
                      action="store", dest="lamda",
                      help="lambda", default=10)
    parser.add_option('-d', '--dataset',
                      action="store", dest="ds",
                      help="Dataset", default="v0.5")
    parser.add_option('-t', '--test',
                      action="store", dest="test",
                      help="Test", default=False)
    hyperparams = parser.parse_args()
    # TRAIN
    main(hyperparams)
