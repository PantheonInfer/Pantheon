from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .urbansound import UrbanSound

INPUT_SIZE = [1, 64, 44]
NUM_CLASSES = 10


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args

    def setup(self, stage):
        transform = T.Compose([T.ToTensor()])
        if stage == 'fit':
            urbansound_full = UrbanSound(self.hparams.data_dir, train=True, transform=transform)
            self.urbansound_train, self.urbansound_val = random_split(urbansound_full, [6500, len(urbansound_full) - 6500])
        if stage == 'test':
            self.urbansound_test = UrbanSound(self.hparams.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.urbansound_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.urbansound_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.urbansound_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )