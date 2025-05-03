import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=24, features=[64, 128, 256]):
        super(UNetClassifier, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, stride=2))
            self.up_convs.append(DoubleConv(feature*2, feature))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features[0], num_classes)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:])
            x = torch.cat((skips[i], x), dim=1)
            x = self.up_convs[i](x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

def load_model(weights_path="unet_weights.pth"):
    model = UNetClassifier()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model
