"""Attention-Based TCN (ATCN) for supply chain delay prediction."""
import torch
import torch.nn as nn
from tcn import TCN


class SelfAttention(nn.Module):
    """Self-attention layer over temporal features."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        """x: (batch, seq_len, hidden)"""
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn_weights, V)


class ATCN(nn.Module):
    """Attention-based Temporal Convolutional Network for delay prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=4, output_size=1):
        super().__init__()
        self.tcn = TCN(input_size, hidden_size, num_layers)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """x: (batch, features, seq_len)"""
        tcn_out = self.tcn(x)  # (batch, hidden, seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, hidden)
        attn_out = self.attention(tcn_out)  # (batch, seq_len, hidden)
        # Use last timestep
        out = attn_out[:, -1, :]
        return self.fc(out)
