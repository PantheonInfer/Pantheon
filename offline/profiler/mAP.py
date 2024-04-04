import torch
from torch import nn
import torch.nn.functional as F
import tempfile
import os
from tqdm import tqdm
import cv2
import sys
import xml.etree.ElementTree as ET

sys.path.append('..')
from third_party.ssd.vision.ssd.config import mobilenetv1_ssd_config
from third_party.ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
from third_party.ssd.vision.utils import box_utils
from third_party.Object_Detection_Metrics.pascalvoc import pascalvoc


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.priors = mobilenetv1_ssd_config.priors
        self.center_variance = mobilenetv1_ssd_config.center_variance
        self.size_variance = mobilenetv1_ssd_config.size_variance
        self.model = model

    def forward(self, x):
        confidences, locations = self.model(x)
        confidences = F.softmax(confidences, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            locations, self.priors, self.center_variance, self.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        return confidences, boxes


def detect(model, orig_img, viz=False):
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='soft')
    boxes, labels, probs = predictor.predict(img, 10, 0.5)
    if viz:
        color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), color[i_color], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        color[i_color],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )


def detect_VOC(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'bicycle', 'bus', 'car', 'train', 'motorbike']
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='soft')
    boxes, labels, probs = predictor.predict(img, 10, 0.1)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            label = classes[labels[i]] + f': {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        [0, 255, 0],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )


def detect_FDDB(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'face']
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='soft')
    boxes, labels, probs = predictor.predict(img, 10, 0.1)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            label = classes[labels[i]] + f': {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        [0, 255, 0],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )

def detect_Fire(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'smoke']
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='soft')
    boxes, labels, probs = predictor.predict(img, 10, 0.1)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            label = classes[labels[i]] + f': {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        [0, 255, 0],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )


def detect_car(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'car']
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    # predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='hard', device=torch.device('cuda'))
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='hard')
    boxes, labels, probs = predictor.predict(img, 10, 0.2)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            label = classes[labels[i]] + f': {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        [0, 255, 0],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )


def detect_light(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'red', 'yellow', 'green', 'unlit']
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='hard', device=torch.device('cuda'))
    boxes, labels, probs = predictor.predict(img, 10, 0.1)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            label = classes[labels[i]] + f': {probs[i]:.2f}'

            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        [0, 255, 0],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        # cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )


def detect_traffic(model, orig_img, viz=False):
    classes = ['BACKGROUND', 'car', 'red', 'yellow', 'green', 'unlit']
    colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 255], [0, 255, 0], [[0, 0, 255]]]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    # predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='hard', device=torch.device('cuda'))
    predictor = create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200, nms_method='hard')
    boxes, labels, probs = predictor.predict(img, 10, 0.2)
    if viz:
        # color = [[], [0, 255, 0], [0, 0, 255]]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
            # label = classes[labels[i]] + f': {probs[i]:.2f}'
            label = classes[labels[i]]
            i_color = int(labels[i])
            box = [round(b.item()) for b in box]

            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), colors[labels[i]], 2)

            cv2.putText(orig_img, label,
                        (box[0] - 0, box[1] - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        colors[labels[i]],
                        2)  # line type

        # print(orig_image.shape)
        cv2.imshow('result', orig_img)
        cv2.imwrite('demo.png', orig_img)
        cv2.waitKey()
    else:
        return dict(
            boxes=boxes,
            scores=probs,
            labels=labels
        )



def annotation(root, id):
    classes = ['BACKGROUND', 'greenlight', 'redlight']
    boxes = []
    labels = []
    with open(os.path.join(root, f'Annotations/{id}.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    for line in lines:
        label, x1, y1, x2, y2 = line.split(' ')
        labels.append(classes.index(label))
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )

def annotation_car(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'car']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in classes:
            is_difficult_str = object.find('difficult').text
            is_difficult = int(is_difficult_str) if is_difficult_str else 0
            if is_difficult:
                continue
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )

def annotation_light(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'red', 'yellow', 'green', 'unlit']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in classes:
            is_difficult_str = object.find('difficult').text
            is_difficult = int(is_difficult_str) if is_difficult_str else 0
            if is_difficult:
                continue
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )

def annotation_traffic(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'car', 'red', 'yellow', 'green', 'unlit']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in classes:
            is_difficult_str = object.find('difficult').text
            is_difficult = int(is_difficult_str) if is_difficult_str else 0
            if is_difficult:
                continue
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )


def annotation_VOC(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'bicycle', 'bus', 'car', 'train', 'motorbike']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in classes:
            is_difficult_str = object.find('difficult').text
            is_difficult = int(is_difficult_str) if is_difficult_str else 0
            if is_difficult:
                continue
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )


def annotation_FDDB(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'face']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list

        bbox = object.find('bndbox')

        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        boxes.append([x1, y1, x2, y2])

        labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )


def annotation_Fire(root, id):
    annotation_file = os.path.join(root, f'Annotations/{id}.xml')
    objects = ET.parse(annotation_file).findall('object')
    boxes = []
    labels = []
    classes = ['BACKGROUND', 'smoke']
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list

        bbox = object.find('bndbox')

        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        boxes.append([x1, y1, x2, y2])

        labels.append(classes.index(class_name))

    return dict(
        boxes=torch.tensor(boxes),
        labels=torch.tensor(labels)
    )


def eval_mAP(model, data_dir):
    classes = ['BACKGROUND', 'greenlight', 'redlight']
    model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        for image_set in ['ll', 'hl']:
            root = os.path.join(data_dir, 'indoor_smart_traffic_dataset', image_set)
            imageset = os.path.join(data_dir, 'indoor_smart_traffic_dataset', image_set, 'ImageSets\Main\\test.txt')
            ids = []
            with open(imageset) as f:
                for line in f:
                    ids.append(line.rstrip())
            for id in tqdm(ids):
                img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
                img = cv2.imread(img_path)
                preds = detect(model, img)
                targets = annotation(root, id)
                with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                    for i in range(len(targets['boxes'])):
                        f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                          int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                          int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

                with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                    for i in range(len(preds['boxes'])):
                        f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                             float(preds['scores'][i]),
                                                             int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                             int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_VOC(model, data_dir):
    classes = ['BACKGROUND', 'bicycle', 'bus', 'car', 'train', 'motorbike']
    model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'VOC2007_vehicle_detection')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_VOC(model, img)
            targets = annotation_VOC(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_FDDB(model, data_dir):
    classes = ['BACKGROUND', 'face']
    model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'FDDB')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_FDDB(model, img)
            targets = annotation_FDDB(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_Fire(model, data_dir):
    classes = ['BACKGROUND', 'smoke']
    model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'Wildfire_Smoke')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_Fire(model, img)
            targets = annotation_Fire(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_car(model, data_dir):
    classes = ['BACKGROUND', 'car']
    # model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'traffic_dataset_car')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_car(model, img)
            targets = annotation_car(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_traffic(model, data_dir):
    classes = ['BACKGROUND', 'car', 'red', 'yellow', 'green', 'unlit']
    # model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'traffic_dataset_vocformat')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_traffic(model, img)
            targets = annotation_traffic(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def eval_mAP_light(model, data_dir):
    classes = ['BACKGROUND', 'red', 'yellow', 'green', 'unlit']
    # model = Wrapper(model)
    model = model.cuda()
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        gtroot = os.path.join(temp_dir, 'groundtruths')
        detroot = os.path.join(temp_dir, 'detections')
        os.mkdir(gtroot)
        os.mkdir(detroot)
        root = os.path.join(data_dir, 'traffic_dataset_light')
        imageset = os.path.join(root, 'ImageSets\\Main\\test.txt')
        ids = []
        with open(imageset) as f:
            for line in f:
                ids.append(line.rstrip())
        for id in tqdm(ids):
            img_path = os.path.join(root, f'JPEGImages\\{id}.jpg')
            img = cv2.imread(img_path)
            preds = detect_light(model, img)
            targets = annotation_light(root, id)
            with open(os.path.join(gtroot, f'{id}.txt'), 'w') as f:
                for i in range(len(targets['boxes'])):
                    f.write('{} {} {} {} {}\n'.format(classes[targets['labels'][i]],
                                                      int(targets['boxes'][i][0]), int(targets['boxes'][i][1]),
                                                      int(targets['boxes'][i][2]), int(targets['boxes'][i][3])))

            with open(os.path.join(detroot, f'{id}.txt'), 'w') as f:
                for i in range(len(preds['boxes'])):
                    f.write('{} {} {} {} {} {}\n'.format(classes[preds['labels'][i]],
                                                         float(preds['scores'][i]),
                                                         int(preds['boxes'][i][0]), int(preds['boxes'][i][1]),
                                                         int(preds['boxes'][i][2]), int(preds['boxes'][i][3])))

        return pascalvoc(gtroot, detroot)


def viz(orig_img, boxes, labels, classes=['BACKGROUND', 'car']):

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        # label = [f'green light: {probs[i]:.2f}' if labels[i] == 1 else f'red light: {probs[i]:.2f}']
        label = classes[labels[i]]

        i_color = int(labels[i])
        box = [round(b.item()) for b in box]

        cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)

        cv2.putText(orig_img, label,
                    (box[0] - 10, box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    [0, 255, 0],
                    2)  # line type

    # print(orig_image.shape)
    cv2.imshow('result', orig_img)
    # cv2.imwrite('demo.png', orig_img)
    cv2.waitKey()


if __name__ == '__main__':
    from dnn.object_detection.ssdlite import ssdlite
    from common.load_state_dict import load_state_dict

    # model = ssdlite(3)
    # model.load_state_dict(load_state_dict(
    #     r'D:\dev\Pantheon\experiments\logs\training\pretrain\object_detection\version_0\checkpoints\epoch=49-step=4749.ckpt'))
    # img = cv2.imread(r'C:\Users\lxhan2\data\indoor_smart_traffic_dataset\scene5\scene5_hl_imgs\scene5_hl_img_50.jpg')
    # wrapped = Wrapper(model)
    # detect(wrapped, img, True)

    # model = ssdlite(6)
    # model.load_state_dict(load_state_dict(
    #     r'D:\dev\Pantheon\experiments\logs\training\pretrain\vehicle_detection\version_0\checkpoints\epoch=99-step=6199.ckpt'))
    # img = cv2.imread(r'C:\Users\lxhan2\data\VOC2007_vehicle_detection\JPEGImages\000333.jpg')
    # wrapped = Wrapper(model)
    # # detect_VOC(wrapped, img, True)
    #
    # print(eval_mAP_VOC(model, r'C:\Users\lxhan2\data'))

    # model = ssdlite(2)
    # model.load_state_dict(load_state_dict(
    #     r'D:\dev\Pantheon\experiments\logs\training\pretrain\face_detection\version_0\checkpoints\epoch=99-step=8199.ckpt'))
    # # img = cv2.imread(r'C:\Users\lxhan2\data\VOC2007_vehicle_detection\JPEGImages\000333.jpg')
    # wrapped = Wrapper(model)
    # # detect_VOC(wrapped, img, True)
    #
    # print(eval_mAP_FDDB(model, r'C:\Users\lxhan2\data'))

    # model = ssdlite(2)
    # model.load_state_dict(load_state_dict(
    #     r'D:\dev\Pantheon\experiments\logs\training\pretrain\wildfire_detection\version_0\checkpoints\epoch=99-step=2699.ckpt'))
    # # img = cv2.imread(r'C:\Users\lxhan2\data\VOC2007_vehicle_detection\JPEGImages\000333.jpg')
    # wrapped = Wrapper(model)
    # # detect_VOC(wrapped, img, True)
    #
    # print(eval_mAP_Fire(model, r'C:\Users\lxhan2\data'))


    # def export2jit(model, dummy_input, save_path):
    #     model.eval()
    #     model_jit = torch.jit.trace(model, dummy_input)
    #     torch.jit.save(model_jit, save_path)
    # export2jit(model, torch.randn([1, 3, 300, 300]), 'model_0.pt')

    # img = cv2.imread(r'C:\Users\lxhan2\data\traffic_dataset_car\JPEGImages\record_2023_11_20_23_04_25_frame_1590.jpg')
    # annot = annotation_car(r'C:\Users\lxhan2\data\traffic_dataset_car', 'record_2023_11_20_23_04_25_frame_1590')
    # viz(img, annot['boxes'], annot['labels'])

    model = ssdlite(6)
    model.load_state_dict(load_state_dict(
        r'D:\dev\Pantheon\experiments\logs\training\pretrain\traffic_detection\version_0\checkpoints\epoch=199-step=5999.ckpt'))
    img = cv2.imread(r'C:\Users\lxhan2\data\traffic_dataset_car\JPEGImages\record_2023_11_20_22_47_27_frame_2835.jpg')
    wrapped = Wrapper(model)
    detect_traffic(wrapped, img, True)
    # print(eval_mAP_car(model, r'C:\Users\lxhan2\data'))