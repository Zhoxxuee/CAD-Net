import torch
import torch.nn as nn

class Cnn1d(nn.Module):
    def __init__(self, num_classes=12, image_width=28):
        super(Cnn1d, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),

            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=88 * 64, out_features=1024),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cnn1d(model_path, pretrained=False, **kwargs):
    model = Cnn1d(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    return model


