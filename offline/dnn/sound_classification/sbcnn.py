from torch import nn


class SBCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, 1),
            nn.BatchNorm2d(24),
            nn.MaxPool2d((4, 2), (4, 2)),
            nn.ReLU(True),
            nn.Conv2d(24, 48, 5, 1),
            nn.BatchNorm2d(48),
            nn.MaxPool2d((4, 2), (4, 2)),
            nn.ReLU(True),
            nn.Conv2d(48, 48, 5, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2400, 64),
            nn.ReLU(True),
            nn.Linear(64, 10)
        )
        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchinfo import summary
    sbcnn = SBCNN()
    summary(sbcnn, (1, 1, 128, 128), device='cpu')