import sys
sys.path.append('../..')

from third_party.ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

def ssdlite(num_classes, train=True):
    return create_mobilenetv2_ssd_lite(num_classes, is_test=not train)


if __name__ == '__main__':
    import torch
    model = ssdlite(2)
    dummy_input = torch.randn((2, 3, 300, 300))
    y = model(dummy_input)
    print(y[0].shape, y[1].shape)