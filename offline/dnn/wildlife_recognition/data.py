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


class OregonWildlife(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.image_list = []
        self.label_list = []
        self.root_dir = root_dir
        self.transform = transform

        lines = []
        if train:
            with open(os.path.join(root_dir, 'trainval_list.txt'), 'r') as f:
                lines = f.readlines()
        else:
            with open(os.path.join(root_dir, 'test_list.txt'), 'r') as f:
                lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            img_path = os.path.join(self.root_dir, line[0])
            label = int(line[1])
            image = Image.open(img_path).convert('RGB')
            image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[2]))
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
# class OregonWildlife(Dataset):
#     def __init__(self, root_dir, transform=None, train=True):
#         self.image_list = []
#         self.label_list = []
#         self.root_dir = root_dir
#         self.transform = transform
#
#         lines = []
#         if train:
#             with open(os.path.join(root_dir, 'trainval_list.txt'), 'r') as f:
#                 lines = f.readlines()
#         else:
#             with open(os.path.join(root_dir, 'test_list.txt'), 'r') as f:
#                 lines = f.readlines()
#         for line in lines:
#             line = line.split(' ')
#             img_path = os.path.join(self.root_dir, line[0])
#             label = int(line[1])
#             self.image_list.append(img_path)
#             self.label_list.append(label)
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         img_path = self.image_list[idx]
#         image = Image.open(img_path).convert('RGB')
#         image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[2]))
#         label = self.label_list[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, torch.tensor(label)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args

    def setup(self, stage):
        if stage == 'fit':
            transform = T.Compose(
                [
                    # T.Resize(INPUT_SHAPE[1]),
                    T.RandomCrop(INPUT_SHAPE[1], padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ]
            )
            dataset = OregonWildlife(os.path.join(self.hparams.data_dir, 'oregon_wildlife'), train=True, transform=transform)
            self.dataset_train, self.dataset_val = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        if stage == 'test':
            transform = T.Compose(
                [
                    # T.Resize(INPUT_SHAPE[1]),
                    T.ToTensor(),
                ]
            )
            self.dataset_test = OregonWildlife(os.path.join(self.hparams.data_dir, 'oregon_wildlife'), train=False, transform=transform)

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
