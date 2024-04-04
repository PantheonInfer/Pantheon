from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import importlib
import argparse
from lit_module import BasicLitClassifier, BasicLitDetector

import sys
sys.path.append('../..')
# from third_party.ssd.vision.ssd.config import mobilenetv1_ssd_config
# from third_party.ssd.vision.ssd.ssd import MatchPrior
from third_party.ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform


def main(args):
    seed_everything(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    logger = TensorBoardLogger(args.log, name=args.task)
    checkpoint = ModelCheckpoint(monitor='acc/val', mode='max', save_last=False, save_weights_only=True)

    trainer = Trainer(
        logger=logger,
        gpus=-1,
        # gpus=0,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint if 'detection' not in args.task else True,
        automatic_optimization=False
    )

    if 'detection' not in args.task:
        model = BasicLitClassifier(args)
    else:
        model = BasicLitDetector(args)

    data_lib = importlib.import_module('dnn.{}.data'.format(args.task))
    data = data_lib.DataModule(args)

    trainer.fit(model, data)
    if 'detection' not in args.task:
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        choices=['image_classification', 'object_detection', 'sign_recognition', 'sound_classification',
                                 'vehicle_detection', 'face_detection', 'age_classification', 'gender_classification',
                                 'emotion_classification', 'wildfire_detection', 'wildlife_recognition', 'scene_recognition',
                                 'traffic_detection'])
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\lxhan2\data')
    parser.add_argument('--log', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()
    main(args)