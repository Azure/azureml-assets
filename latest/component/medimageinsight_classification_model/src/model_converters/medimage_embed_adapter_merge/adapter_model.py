# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for a multi-layer perceptron model definition."""

import torch
from torch import nn


class MLP_model(nn.Module):
    """A multi-layer perceptron (MLP) model for processing vision features and making predictions."""

    def __init__(self, in_channels, hidden_dim, num_class, device):
        """
        Initialize the AdapterModel.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Dimension of the hidden layer.
            num_class (int): Number of output classes.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_class = num_class

        # Adaptor Module
        self.vision_embd = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.retrieval_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
        )

        # Prediction Head
        self.prediction_head = nn.Sequential(nn.Linear(self.hidden_dim, self.num_class))

    def forward(self, vision_feat):
        """
        Perform the forward pass for the adapter model.

        Args:
            vision_feat (torch.Tensor): Input tensor containing vision features.

        Returns:
            tuple: A tuple containing:
            - feat (torch.Tensor): The processed feature tensor after embedding and convolution.
            - class_output (torch.Tensor): The output tensor from the prediction head.
        """
        vision_feat = torch.tensor(vision_feat, device=self.device)
        feat = self.vision_embd(vision_feat.squeeze(1))
        feat = self.retrieval_conv(torch.unsqueeze(feat, 2))
        class_output = self.prediction_head(feat.squeeze(2))

        return feat, class_output
