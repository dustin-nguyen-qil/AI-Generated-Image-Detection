import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F

"""
    Compute the fingerprint image from rich texture image and poor texture image
"""
class FingerPrint(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # load 30 high pass filters
        self.kernels = self.load_high_pass_filters()

        self.conv_block = nn.Sequential(
            nn.Conv2d(30, 3, (5, 5)),
            nn.BatchNorm2d(3),
            nn.Hardtanh()
        )

    def forward(self, rich_patch, poor_patch):
        # pass into highpass filters
        rich_filterd = F.conv2d(rich_patch, self.kernels)
        poor_filterd = F.conv2d(poor_patch, self.kernels)

        # get rich texture image
        out_rich = self.conv_block(rich_filterd)
        # get poor texture image
        out_poor = self.conv_block(poor_filterd)
        # get fingerprint image
        fingerprint = torch.sub(out_poor, out_rich)

        return fingerprint
    
    def load_high_pass_filters(self):
        kernels = np.load('SRM_Kernels.npy')
        filters = np.tile(kernels, (1, 1, 3, 1))
        filters_tensors = [torch.tensor(f, dtype=torch.float32).permute(2, 0, 1) for f in np.moveaxis(filters, -1, 0)]
        return torch.stack(filters_tensors)
    
# Test 
if __name__ == "__main__":
    rich = torch.rand(32, 3, 256, 256)
    poor = torch.rand(32, 3, 256, 256)
    fingerprint = FingerPrint()
    print(fingerprint(rich, poor).size())