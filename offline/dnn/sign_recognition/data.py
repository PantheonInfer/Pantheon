from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .gtsrb import GTSRB

INPUT_SIZE = [3, 32, 32]
NUM_CLASSES = 43


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args
        self.mean = (0.3337, 0.3064, 0.3171)
        self.std = (0.2672, 0.2564, 0.2629)

    def setup(self, stage):
        if stage == 'fit':
            transform = T.Compose(
                [
                    T.Resize((32, 32)),
                    T.RandomCrop(32, padding=4),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            gtsrb_full = GTSRB(self.hparams.data_dir, train=True, transform=transform)
            self.gtsrb_train, self.gtsrb_val = random_split(gtsrb_full, [35000, len(gtsrb_full) - 35000])
        if stage == 'test':
            transform = T.Compose(
                [
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
            self.gtsrb_test = GTSRB(self.hparams.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.gtsrb_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.gtsrb_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.gtsrb_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )
