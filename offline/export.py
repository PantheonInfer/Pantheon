import os

import torch
import argparse
import importlib
import copy
from een.een import EEN
from common.load_state_dict import load_state_dict


def export2jit(model, dummy_input, save_path):
    model.eval()
    model_jit = torch.jit.trace(model, dummy_input)
    torch.jit.save(model_jit, save_path)


def export(args):
    configs = importlib.import_module(f'dnn.{args.task}.configs')
    dummy_input = torch.randn([1] + configs.INPUT_SHAPE)
    if 'detection' not in args.task:
        if args.task == 'image_classification':
            model_lib = importlib.import_module('dnn.image_classification.resnet')
            model = model_lib.resnet50()
        elif args.task == 'sign_recognition':
            model_lib = importlib.import_module('dnn.sign_recognition.mobilenet_v2')
            model = model_lib.mobilenet_v2(num_classes=43)
        elif args.task == 'sound_classification':
            model_lib = importlib.import_module('dnn.sound_classification.sbcnn')
            model = model_lib.SBCNN()
        elif args.task == 'age_classification':
            model_lib = importlib.import_module('dnn.age_classification.resnet')
            model = model_lib.resnet34()
        elif args.task == 'gender_classification':
            model_lib = importlib.import_module('dnn.gender_classification.vgg')
            model = model_lib.vgg16_bn()
        elif args.task == 'emotion_classification':
            model_lib = importlib.import_module('dnn.emotion_classification.resnet')
            model = model_lib.resnet50()
        elif args.task == 'wildlife_recognition':
            model_lib = importlib.import_module('dnn.wildlife_recognition.googlenet')
            model = model_lib.googlenet()
        elif args.task == 'scene_recognition':
            model_lib = importlib.import_module('dnn.scene_recognition.resnet')
            model = model_lib.resnet18()
        model = EEN(model, dummy_input, args.task, num_classes=configs.NUM_CLASSES)
    else:
        model_lib = importlib.import_module('dnn.object_detection.ssdlite')
        if args.task == 'object_detection':
            model = model_lib.ssdlite(num_classes=3)
        elif args.task == 'vehicle_detection':
            model = model_lib.ssdlite(num_classes=6)
        elif args.task == 'face_detection':
            model = model_lib.ssdlite(num_classes=2)
        elif args.task == 'wildfire_detection':
            model = model_lib.ssdlite(num_classes=2)
        model = EEN(model, dummy_input, args.task, num_classes=configs.NUM_CLASSES, end_node='base_net.14.conv.0.2')
    try:
        model.load_state_dict(args.weights)
    except:
        model.load_state_dict(load_state_dict(args.weights))
    model.eval()

    num = len(model.blocks)
    x = copy.deepcopy(dummy_input)
    os.makedirs(args.save)
    with torch.no_grad():
        for i in range(num):
            export2jit(model.blocks[i], x, os.path.join(args.save, 'block_{:02d}.pth'.format(i)))
            x = model.blocks[i](x)
            export2jit(model.branches[i], x, os.path.join(args.save, 'branch_{:02d}.pth'.format(i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str)
    # parser.add_argument('--task', type=str)
    # parser.add_argument('--save', type=str)
    parser.add_argument('--weights', type=str, default='../experiments/logs/training/construct/vehicle_detection/version_0/checkpoints/epoch=99-step=6199.ckpt')
    parser.add_argument('--task', type=str, default='vehicle_detection')
    parser.add_argument('--save', type=str, default='../experiments/weights/vehicle_detection')
    args = parser.parse_args()
    export(args)

