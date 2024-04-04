import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import os
from PIL import Image
from .configs import INPUT_SHAPE


class AdienceGender(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.image_list = []
        self.label_list = []
        self.root_dir = root_dir
        self.transform = transform
        if train:
            with open(os.path.join(root_dir, 'gendertrainval.txt'), 'r') as f:
                lines = f.readlines()
        else:
            with open(os.path.join(root_dir, 'gendertest.txt'), 'r') as f:
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


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def setup(self, stage):
        if stage == 'fit':
            transform = T.Compose(
                [
                    # T.Resize(INPUT_SHAPE[1]),
                    T.RandomCrop(INPUT_SHAPE[1], padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            dataset = AdienceGender(os.path.join(self.hparams.data_dir, 'Adience'), train=True, transform=transform)
            self.dataset_train, self.dataset_val = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        if stage == 'test':
            transform = T.Compose(
                [
                    # T.Resize(INPUT_SHAPE[1]),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            self.dataset_test = AdienceGender(os.path.join(self.hparams.data_dir, 'Adience'), train=False, transform=transform)

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
