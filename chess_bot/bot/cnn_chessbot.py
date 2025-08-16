from torchvision.models import resnet18
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from typing import Type


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


class ChessPolicyNet(nn.Module):
    """
    A deep ResNet-based model for a chess policy head, inspired by AlphaZero.

    This network takes a 12-channel, 8x8 representation of a chessboard
    and outputs a 4096-element vector representing the probability
    distribution over all possible 64 'from' to 64 'to' squares.
    """

    def __init__(
        self, num_input_channels: int = 12, num_blocks: int = 8, num_channels: int = 128
    ):
        """
        Initializes the ChessPolicyNet.

        Args:
            num_input_channels: Number of input channels for the board state.
            num_blocks: The number of residual blocks in the network body.
                        More blocks increase the network's depth and capacity.
            num_channels: The number of channels used throughout the convolutional
                          layers. More channels increase the network's width.
        """
        super().__init__()

        # 1. Initial convolution: Preserves the 8x8 dimension
        self.conv1 = nn.Conv2d(
            num_input_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Body of the network: A series of residual blocks
        self.layers = self._make_layer(BasicBlock, num_channels, num_blocks)

        # 3. Policy Head: This part generates the move probabilities
        self.policy_head_fc = nn.Linear(num_channels * 8 * 8, 4096)

    def _make_layer(
        self, block: Type[BasicBlock], channels: int, num_blocks: int
    ) -> nn.Sequential:
        """Creates a sequence of residual blocks."""
        layers = []
        for _ in range(num_blocks):
            # In BasicBlock, the input and output channels are the same
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x: The input tensor of shape (batch_size, 12, 8, 8).

        Returns:
            The output tensor of shape (batch_size, 4096).
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        x = self.layers(x)

        # --- Policy Head ---
        # Flatten the output from (batch, channels, 8, 8) to (batch, channels * 64)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Apply the final fully connected layer
        policy = self.policy_head_fc(x)

        return policy
