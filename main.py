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
from pytorch_lightning.loggers import WandbLogger


def main(hparams):
    bottelNeck = int(hparams[0].hidden)
    lamda = float(hparams[0].lamda)
    ds = hparams[0].ds
    fun = int(hparams[0].fun)
    test = bool(hparams[0].test)
    print(bottelNeck, lamda, ds, test)
    datamodule = DrivingDataMadule(ds, 5800, 176, 10000)
    wandb_logger = WandbLogger()
    if not test:
        model = Siamese(battle_neck=bottelNeck, lamda=lamda)
        checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss',
            filename='LSTMEncoderLSTM--{v_num:02d}-{epoch:02d}-{validation_loss:.9f}-{similarity_loss:.15f}',
        )
        early_callback = EarlyStopping(monitor='validation_loss')
        trainer = pl.Trainer(gpus=-1, max_epochs=1000,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback],
                             num_nodes=1)
        trainer.fit(model=model, datamodule=datamodule)
    else:
        for filename in os.listdir("checkpoint"):
            if filename in ["b20l10d1"]:
                for c in os.listdir(f'checkpoint/{filename}/checkpoints'):
                    print(f'{filename}/{c}')
                    bottelNeck = int(re.search(r"b(.*?)l", filename).group(1))
                    lamda = float(re.search(r"l(.*?)d", filename).group(1))
                    # ds = re.search(r"d(.*?).", filename)

                    model = Siamese.load_from_checkpoint(
                        f'checkpoint/{filename}/checkpoints/{c}',
                        battle_neck=bottelNeck, lamda=lamda, fun=fun
                    )

                    trainer = pl.Trainer(gpus=-1, max_epochs=100, accelerator='dp',
                                         callbacks=[LSTMCallback(name=f'{ds}-{fun}')],
                                         num_nodes=1)
                    trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())


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
    parser.add_option('-f', '--function',
                      action="store", dest="fun",
                      help="Loss Function", default=1)
    hyperparams = parser.parse_args()
    # TRAIN
    main(hyperparams)
