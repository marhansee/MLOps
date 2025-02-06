import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove final layer

        # Freeze the inner layers (e.g., layers before 'layer4')
        for name, param in self.resnet.named_parameters():
            if not name.startswith("layer1") and not name.startswith("layer2"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),  # BatchNorm after Linear layer
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # BatchNorm after Linear layer
            nn.ReLU(),
            nn.Linear(512, 196),  # Assuming 196 classes
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x
