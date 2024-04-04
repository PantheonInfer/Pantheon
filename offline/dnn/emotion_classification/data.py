import numpy as np
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import os
from PIL import Image
import pandas as pd
from .configs import INPUT_SHAPE


class FER(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.image_list = []
        self.label_list = []
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(os.path.join(root_dir, 'fer2013/fer2013.csv'))
        if train:
            df = df[df['Usage'] == 'Training']
        else:
            df = df[df['Usage'] == 'PrivateTest']
        for index, row in df.iterrows():
            px = row.pixels
            image = np.array(px.split(' ')).reshape(48, 48, 1).astype('float32')
            label = int(row.emotion)
            self.image_list.append(image)
            self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args

    def setup(self, stage):
        if stage == 'fit':
            transform = T.Compose(
                [
                    T.ToPILImage(),
                    # T.Resize(INPUT_SHAPE[1]),
                    T.RandomCrop(INPUT_SHAPE[1], padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ]
            )
            dataset = FER(os.path.join(self.hparams.data_dir, 'FER'), train=True, transform=transform)
            self.dataset_train, self.dataset_val = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        if stage == 'test':
            transform = T.Compose(
                [
                    T.ToPILImage(),
                    # T.Resize(INPUT_SHAPE[1]),
                    T.ToTensor(),
                ]
            )
            self.dataset_test = FER(os.path.join(self.hparams.data_dir, 'FER'), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )
