from fingerprint import FingerPrint
from classifier import Classifier
from torch import nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Create fingerprint from the input image
        self.fingerprint = FingerPrint()
        
        self.classifier = Classifier()

    def forward(self, rich_image, poor_image):
        fingerprint = self.fingerprint(rich_image, poor_image)
        # Pass the fingerprint through the classifier
        return self.classifier(fingerprint)