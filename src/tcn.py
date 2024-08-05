"""Temporal Convolutional Network with dilated causal convolutions."""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Causal convolution: output at time t depends only on inputs at time <= t."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                          padding=self.padding, dilation=dilation))

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # Remove future padding


class TCNBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return self.relu(out + self.residual(x))


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation."""

    def __init__(self, input_size, hidden_size=64, num_layers=4, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, features, seq_len)"""
        return self.network(x)
