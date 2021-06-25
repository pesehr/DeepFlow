import os
from torch.utils.data import Dataset
from data_module.XML_parrser import XMLParser
from torch import tensor, device, mean, std
from torch.utils.data import DataLoader
import pytorch_lightning as pl

'''
 Creates and loads the Driving dataset
 address: dataset address
 num_objects: number of cars
 start: starting point of the dataset
 end: end point of the dataset
 d: device (Default uses GPU) TODO check if GPU exists 
'''


class DrivingDataset(Dataset):

    def __init__(self, address, start=0, end=10000, num_objects=5, d=device('cuda'),
                 validation=False):
        self.parser = XMLParser(address)
        self.start = start
        self.end = end
        self.num_objects = num_objects
        self.device = d
        self.set = []

        # Params Config
        self.max_x = 965.07
        self.min_x = -41.12
        self.max_y = 50.60
        self.min_y = 44.20

        self.validation = validation
        self.load_set()

    # normalizes the dataset
    def normalize(self, d, max, min):
        return (d - min) / (max - min)

    # reads file and extract the vehicle groups
    def load_set(self):
        raw_data, labels = self.parser.read_txt()
        b = []
        for r in range(0, len(raw_data) - self.num_objects + 1, self.num_objects):
            b.append([tensor(raw_data[k], device=self.device) for k in range(r, r + self.num_objects)])
            label = [labels[k] for k in range(r, r + self.num_objects)]
            self.set.append((b, label))
            b = []

    def __getitem__(self, index):
        if self.validation:
            return self.set[10 * index]
        else:
            return self.set[index]

    def __len__(self):
        if self.validation:
            return int(len(self.set) / 10)
        else:
            return int(len(self.set))


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len, observe_len=5, label_len=1):
        super().__init__()
        self.train_dataset_address = os.path.realpath('.') + f'/dataset/train/{version}'
        self.test_dataset_address = os.path.realpath('.') + f'/dataset/test/{version}'
        self.validation_dataset_address = os.path.realpath('.') + f'/dataset/train/normal'
        self.train_len = train_len
        self.validate_len = validate_len
        self.test_len = test_len
        self.observe_len = observe_len
        self.label_len = label_len

    def train_dataloader(self):
        return DataLoader(DrivingDataset(address=self.train_dataset_address, start=0, end=self.train_len, ),
                          shuffle=True, num_workers=0, batch_size=1)

    def val_dataloader(self):
        return DataLoader(
            DrivingDataset(address=self.validation_dataset_address, start=0,
                           end=self.validate_len, validation=True),
            shuffle=False, num_workers=0, batch_size=1)

    def test_dataloader(self):
        return DataLoader(
            DrivingDataset(address=self.test_dataset_address, start=0, end=self.test_len, ),
            shuffle=False, num_workers=0, batch_size=1)
