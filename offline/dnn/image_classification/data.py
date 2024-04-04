from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage):
        if stage == 'fit':
            transform = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            cifar10_full = CIFAR10(self.hparams.data_dir, train=True, transform=transform, download=True)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])
        if stage == 'test':
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            self.cifar10_test = CIFAR10(self.hparams.data_dir, train=False, transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )
