import torch
import torch.nn as nn
from torchvision import models

"""
    Build a classifier that take the fingerprint image as input and output fake or real
"""
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        # Define convolutional layers with BatchNorm and ReLU activation
        self.conv_layers = nn.Sequential(
            # Convolutional layers with BatchNorm and ReLU activation
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )
        
        # Define another set of convolutional layers with BatchNorm and ReLU activation
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )
        
        # Define another set of convolutional layers with BatchNorm and ReLU activation
        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Average pooling
            nn.AvgPool2d(kernel_size=2)
        )
        
        # Define another set of convolutional layers with BatchNorm and ReLU activation
        self.conv_layers4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(32, 1) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_layers2(x)
        x = self.conv_layers3(x)
        x = self.conv_layers4(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))
        return x

# Can try ResNet Classifier
class ResNetClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.classifer = models.resnet50(weights=weights)

        # modify the classification layer to output 2 values fake (0) or real (1)
        self.classifer.fc = nn.Linear(self.classifer.fc.in_features, 1)
    
    def forward(self, x):
        return self.classifer(x)

if __name__ == "__main__":
    fingerprint = torch.rand(2, 3, 248, 248)
    classifier = Classifier()
    print(classifier(fingerprint))


