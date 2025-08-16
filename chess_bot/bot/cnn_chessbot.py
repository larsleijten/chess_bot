from torchvision.models import resnet18
import torch
import torch.nn as nn


class CNNChessbot(nn.Module):
    """
    A chessbot CNN based on a resnet18 model, adapted to output
    """

    def __init__(self, in_chan: int = 12, classes_out: int = 4096):  # (8 * 8)^2
        super(CNNChessbot, self).__init__()
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            dilation=self.model.conv1.dilation,
            groups=self.model.conv1.groups,
            bias=self.model.conv1.bias,
            padding_mode=self.model.conv1.padding_mode,
        )
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, out_features=classes_out
        )

    def forward(self, x):
        return self.model(x)
