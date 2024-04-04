import torch
from torch import nn
import sys

sys.path.append('../..')
from third_party.ssd.vision.nn.mobilenet_v2 import InvertedResidual
from third_party.ssd.vision.ssd.mobilenet_v2_ssd_lite import SeperableConv2d

exit_configs = {
    150: {
        'base_config': [
            # t, c, s
            [1, 96, 2],
            [1, 96, 2],
            [1, 96, 2],
            [1, 96, 2]
        ],
        'base_source_idx': [2, 3]
    },
    75: {
        'base_config': [
            [1, 160, 2],
            [1, 160, 2],
            [1, 160, 2]
        ],
        'base_source_idx': [1, 2]
    },
    38: {
        'base_config': [
            [1, 320, 2],
            [1, 320, 2],
        ],
        'base_source_idx': [0, 1]
    },
    19: {
        'base_config': [
            [1, 512, 2]
        ],
        'base_source_idx': [-1, 0]
    }
}


class ExitBranch(nn.Module):
    def __init__(self, in_channels, base_config, base_source_idx, num_classes=3):
        super(ExitBranch, self).__init__()
        self.num_classes = num_classes
        self.base_config = base_config
        self.base_source_idx = base_source_idx
        base = []
        inp = in_channels
        for t, c, s in self.base_config:
            base.append(InvertedResidual(inp, c, s, t))
            inp = c
        self.base = nn.ModuleList(base)

        self.extras = nn.ModuleList([
            InvertedResidual(inp, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])

        inp = self.base_config[self.base_source_idx[0]][1] if self.base_source_idx[0] != -1 else in_channels
        self.regression_headers = nn.ModuleList([
            SeperableConv2d(in_channels=inp, out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=self.base_config[self.base_source_idx[1]][1], out_channels=6 * 4, kernel_size=3,
                            padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])

        self.classification_headers = nn.ModuleList([
            SeperableConv2d(in_channels=inp, out_channels=6 * num_classes, kernel_size=3,
                            padding=1),
            SeperableConv2d(in_channels=self.base_config[self.base_source_idx[1]][1], out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        sources = []
        locs = []
        confs = []
        for k in range(self.base_source_idx[0] + 1):
            x = self.base[k](x)
        sources.append(x)

        for k in range(self.base_source_idx[0] + 1, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        for k in range(len(self.extras)):
            x = self.extras[k](x)
            sources.append(x)

        for s, r, c in zip(sources, self.regression_headers, self.classification_headers):
            conf = c(s)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1, self.num_classes)
            confs.append(conf)
            loc = r(s)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            locs.append(loc)

        confs = torch.cat(confs, 1)
        locs = torch.cat(locs, 1)

        return confs, locs


def build_exit_branch(input_size, in_channels, num_classes):
    return ExitBranch(in_channels, exit_configs[input_size]['base_config'], exit_configs[input_size]['base_source_idx'],
                      num_classes)
