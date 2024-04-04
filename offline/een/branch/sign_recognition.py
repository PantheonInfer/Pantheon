import torch
from torch import nn
import sys
sys.path.append('../..')
from common.conv_basic import ConvBasic


class ExitBranch(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, channels=256):
        super(ExitBranch, self).__init__()
        if input_size == 32:
            self.features = nn.Sequential(
                ConvBasic(in_channels, channels, 3, 2, 1),
                nn.MaxPool2d(2, 2),
                ConvBasic(channels, channels, 3, 2, 1),
            )
        elif input_size == 16:
            self.features = nn.Sequential(
                ConvBasic(in_channels, channels, 3, 2, 1),
                nn.MaxPool2d(2, 2),
            )
        elif input_size == 8:
            self.features = nn.Sequential(
                ConvBasic(in_channels, channels, 3, 2, 1),
            )
        elif input_size == 4:
            self.features = nn.Sequential(
                ConvBasic(in_channels, channels, 1, 1, 0)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels, num_classes),
        )

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
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def build_exit_branch(input_size, in_channels, num_classes=43):
    return ExitBranch(input_size, in_channels, num_classes)
