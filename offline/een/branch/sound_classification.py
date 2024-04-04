import torch
from torch import nn


class ExitBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExitBranch, self).__init__()
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 25)),
            nn.Flatten(),
            nn.Linear(in_channels * 50, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
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
        return self.cls(x)


def build_exit_branch(input_size, in_channels, num_classes=10):
    return ExitBranch(in_channels, num_classes)
