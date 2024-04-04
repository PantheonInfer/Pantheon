import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import importlib
from dnn.object_detection.ssdlite import ssdlite
from common.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from common.load_state_dict import load_state_dict
from third_party.ssd.vision.nn.multibox_loss import MultiboxLoss
from third_party.ssd.vision.ssd.config import mobilenetv1_ssd_config
from een.een import EEN


class BasicLitClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        if self.hparams.task == 'image_classification':
            model_lib = importlib.import_module('dnn.image_classification.resnet')
            self.model = model_lib.resnet50()
        elif self.hparams.task == 'sign_recognition':
            model_lib = importlib.import_module('dnn.sign_recognition.mobilenet_v2')
            self.model = model_lib.mobilenet_v2(num_classes=43)
        elif self.hparams.task == 'sound_classification':
            model_lib = importlib.import_module('dnn.sound_classification.sbcnn')
            self.model = model_lib.SBCNN()
        elif self.hparams.task == 'age_classification':
            model_lib = importlib.import_module('dnn.age_classification.resnet')
            self.model = model_lib.resnet34()
        elif self.hparams.task == 'gender_classification':
            model_lib = importlib.import_module('dnn.gender_classification.vgg')
            self.model = model_lib.vgg16_bn()
        elif self.hparams.task == 'emotion_classification':
            model_lib = importlib.import_module('dnn.emotion_classification.resnet')
            self.model = model_lib.resnet50()
        elif self.hparams.task == 'wildlife_recognition':
            model_lib = importlib.import_module('dnn.wildlife_recognition.googlenet')
            self.model = model_lib.googlenet()
        elif self.hparams.task == 'scene_recognition':
            model_lib = importlib.import_module('dnn.scene_recognition.resnet')
            self.model = model_lib.resnet18()

    def forward(self, batch):
        x, y = batch
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        opt.zero_grad()
        loss, accuracy = self.forward(batch)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        self.manual_backward(loss, opt)
        opt.step()

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log('acc/test', accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            'scheduler': WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]


class BasicLitDetector(pl.LightningModule):
    def __init__(self, hparams):
        super(BasicLitDetector, self).__init__()
        self.hparams = hparams
        if hparams.task == 'object_detection':
            self.model = ssdlite(num_classes=3)
        elif hparams.task == 'vehicle_detection':
            self.model = ssdlite(num_classes=6)
        elif hparams.task == 'face_detection':
            self.model = ssdlite(num_classes=2)
        elif hparams.task == 'wildfire_detection':
            self.model = ssdlite(num_classes=2)
        elif hparams.task == 'traffic_detection':
            self.model = ssdlite(num_classes=6)
        self.criterion = MultiboxLoss(mobilenetv1_ssd_config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                      center_variance=0.1, size_variance=0.2, device='cuda')

    def forward(self, batch):
        images, boxes, labels = batch
        confidence, locations = self.model(images)
        loss_l, loss_c = self.criterion(confidence, locations, labels, boxes)
        loss = loss_l + loss_c
        return loss, loss_l, loss_c

    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        opt.zero_grad()
        loss, loss_l, loss_c = self.forward(batch)
        self.log('train/loss', loss)
        self.log('train/loss_l', loss_l)
        self.log('train/loss_c', loss_c)
        self.manual_backward(loss, opt)
        opt.step()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
        )
        scheduler = {
            "scheduler": WarmupMultiStepLR(optimizer, warmup_epochs=len(self.train_dataloader()), milestones=[
                int(self.hparams.max_epochs * len(self.train_dataloader()) * 2 / 3),
                int(self.hparams.max_epochs * len(self.train_dataloader()) * 5 / 6)], gamma=0.1),
            "interval": "step",
        }
        return [optimizer], [scheduler]


class LitClassifier(BasicLitClassifier):
    def __init__(self, hparams):
        super().__init__(hparams)
        configs = importlib.import_module('dnn.{}.configs'.format(hparams.task))
        dummy_input = torch.randn([1] + configs.INPUT_SHAPE)
        try:
            self.model.load_state_dict(hparams.pretrain)
        except:
            self.model.load_state_dict(load_state_dict(hparams.pretrain))
        self.model = EEN(self.model, dummy_input, hparams.task, num_classes=configs.NUM_CLASSES)
        self.model.deactivate_branch(-1)
        self.model.freeze_backbone()

    def forward(self, batch):
        images, labels = batch
        predictions_list = self.model(images)
        loss_list = []
        accuracy_list = []
        for predictions in predictions_list:
            loss_list.append(self.criterion(predictions, labels))
            accuracy_list.append(self.accuracy(predictions, labels))
        return loss_list, accuracy_list

    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        opt.zero_grad()
        loss_list, accuracy_list = self.forward(batch)
        for i, (loss, accuracy) in enumerate(zip(loss_list, accuracy_list)):
            self.log('train loss/branch-{}'.format(i), loss)
            self.log('train acc/branch-{}'.format(i), accuracy)
            if i != len(loss_list) - 1:
                self.manual_backward(loss, opt, retain_graph=True)
            else:
                self.manual_backward(loss, opt)
        opt.step()
        loss = torch.sum(torch.tensor(loss_list))
        accuracy = torch.mean(torch.tensor(accuracy_list))
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)

    def validation_step(self, batch, batch_nb):
        loss_list, accuracy_list = self.forward(batch)
        for i, (loss, accuracy) in enumerate(zip(loss_list, accuracy_list)):
            self.log('val loss/branch-{}'.format(i), loss)
            self.log('val acc/branch-{}'.format(i), accuracy)
        loss = torch.sum(torch.tensor(loss_list))
        accuracy = torch.mean(torch.tensor(accuracy_list))
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)

    def test_step(self, batch, batch_nb):
        loss_list, accuracy_list = self.forward(batch)
        for i, (loss, accuracy) in enumerate(zip(loss_list, accuracy_list)):
            self.log('test acc/branch-{}'.format(i), accuracy)
        accuracy = torch.mean(torch.tensor(accuracy_list))
        self.log('acc/test', accuracy)


class LitDetector(BasicLitDetector):
    def __init__(self, hparams):
        super().__init__(hparams)
        configs = importlib.import_module('dnn.{}.configs'.format(hparams.task))
        dummy_input = torch.randn([1] + configs.INPUT_SHAPE)
        try:
            self.model.load_state_dict(hparams.pretrain)
        except:
            self.model.load_state_dict(load_state_dict(hparams.pretrain))
        self.model = EEN(self.model, dummy_input, hparams.task, num_classes=configs.NUM_CLASSES, end_node='base_net.14.conv.0.2')
        self.model.deactivate_branch(-1)
        self.model.freeze_backbone()

    def forward(self, batch):
        images, boxes, labels = batch
        predictions_list = self.model(images)
        loss_l_list = []
        loss_c_list = []
        loss_list = []
        for confidence, locations in predictions_list:
            loss_l, loss_c = self.criterion(confidence, locations, labels, boxes)
            loss = loss_l + loss_c
            loss_l_list.append(loss_l)
            loss_c_list.append(loss_c)
            loss_list.append(loss)
        return loss_list, loss_l_list, loss_c_list

    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        opt.zero_grad()
        loss_list, loss_l_list, loss_c_list = self.forward(batch)
        for i, (loss, loss_l, loss_c) in enumerate(zip(loss_list, loss_l_list, loss_c_list)):
            self.log('exit-{}/loss'.format(i), loss)
            self.log('exit-{}/loss_l'.format(i), loss_l)
            self.log('exit-{}/loss_c'.format(i), loss_c)
            if i != len(loss_list) - 1:
                self.manual_backward(loss, opt, retain_graph=True)
            else:
                self.manual_backward(loss, opt)
        opt.step()

if __name__ == '__main__':
    import sys

    sys.path.append('..')
    # from dnn.image_classification.resnet import resnet50
    # model = resnet50()
    # from dnn.sign_recognition.mobilenet_v2 import mobilenet_v2
    # model = mobilenet_v2()
    # from dnn.sound_classification.sbcnn import SBCNN
    # model = SBCNN()
    from dnn.object_detection.ssdlite import ssdlite
    model = ssdlite(3)
    print(model.config.center_variance)
    # dummy_input = torch.randn((2, 3, 300, 300))
    # # torch.onnx.export(model, dummy_input, 'temp.onnx')
    # task = 'object_detection'
    # num_classes = 3
    # # model = EEN(model, dummy_input, task, num_classes=num_classes)
    # model = EEN(model, dummy_input, task, num_classes=num_classes, end_node='base_net.14.conv.0.2')
    # def cal_total_params(module):
    #     total_params = sum(param.numel() for param in module.parameters())
    #     return total_params
    #
    # # for i in range(len(model.blocks)):
    # #     print('BLOCK:', cal_total_params(model.blocks[i]) * 4 / 1024 / 1024, 'MB')
    # #     print('BRANCH', cal_total_params(model.branches[i]) * 4 / 1024 / 1024, 'MB')
    # model(dummy_input)
    #
