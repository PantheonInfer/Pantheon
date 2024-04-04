import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os

import sys
sys.path.append('../..')
from third_party.ssd.vision.ssd.config import mobilenetv1_ssd_config
from third_party.ssd.vision.ssd.ssd import MatchPrior
from third_party.ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from third_party.ssd.vision.datasets.voc_dataset import VOCDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.hparams = args

    def train_dataloader(self):
        target_transform = MatchPrior(mobilenetv1_ssd_config.priors,
                                      mobilenetv1_ssd_config.center_variance,
                                      mobilenetv1_ssd_config.size_variance, 0.5)
        train_transform = TrainAugmentation(mobilenetv1_ssd_config.image_size,
                                            mobilenetv1_ssd_config.image_mean,
                                            mobilenetv1_ssd_config.image_std)

        self.train_dataset = VOCDataset(os.path.join(self.hparams.data_dir, 'FDDB'), transform=train_transform, target_transform=target_transform, is_test=False)

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )