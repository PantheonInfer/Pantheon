import numpy as np
import os
import sys
sys.path.append('../..')

from third_party.ssd.vision.datasets.voc_dataset import VOCDataset


class IndoorSmartTraffic(VOCDataset):
    base_folder = 'indoor_smart_traffic_dataset'
    def __init__(self, root_dir, _set, transform=None, target_transform=None, train=True):
        super(IndoorSmartTraffic, self).__init__(os.path.join(root_dir, self.base_folder, _set), transform, target_transform, is_test=not train)
        self.class_names = ['BACKGROUND', 'greenlight', 'redlight']
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.root, f"Annotations/{image_id}.txt")

        boxes = []
        labels = []
        is_difficult = []
        with open(annotation_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            label, x1, y1, x2, y2 = line.split(' ')
            labels.append(self.class_dict[label])
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            is_difficult.append(0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))


if __name__ == '__main__':
    import cv2

    dataset = IndoorSmartTraffic(r'C:\Users\lxhan2\data', 'll')
    colors = {'greenlight': (0, 255, 0), 'redlight': (0, 0, 255)}
    for i in range(100):
        image, boxes, labels = dataset.__getitem__(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            label = dataset.class_names[labels[i]]
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
            cv2.putText(image, label,
                        (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        colors[label],
                        2)  # line type
        cv2.imshow('Image', image)
        cv2.waitKey()