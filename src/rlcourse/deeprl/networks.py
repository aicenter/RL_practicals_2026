"""Neural-network building blocks shared by the deep-RL weeks.

`GridEncoder` turns a batch of PragMan boards into feature vectors. Week 7 puts a Q-value head
on top of it, Weeks 8 and 9 a policy head (and a value head), so the part of the network that
*looks at the board* stays the same while the algorithm around it changes.

    from rlcourse.deeprl.networks import GridEncoder

    encoder = GridEncoder(env.observation_space.shape)
    head = torch.nn.Linear(encoder.n_features, n_actions)
    q_values = head(encoder(observations))            # observations: (batch, H, W, C), uint8
"""

from __future__ import annotations

import torch
from torch import nn


class GridEncoder(nn.Module):
    """Board planes (batch, H, W, C) -> features (batch, n_features).

    Two 3x3 convolutions followed by a fully-connected layer. A convolution applies the same
    small filter at every cell, which is how we tell the network that a ghost next to the agent
    means the same thing anywhere on the board. With padding, the board keeps its size, and after
    two layers each cell has seen its 5x5 neighbourhood; the fully-connected layer then combines
    the whole board.
    """

    def __init__(self, observation_shape: tuple[int, ...], channels: int = 16, n_features: int = 128):
        super().__init__()
        height, width, in_channels = observation_shape
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * height * width, n_features),
            nn.ReLU(),
        )
        self.n_features = n_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # PragMan boards are channel-last and uint8 (0 or 1); convolutions want channel-first floats.
        x = observations.float().permute(0, 3, 1, 2)
        return self.layers(x)
