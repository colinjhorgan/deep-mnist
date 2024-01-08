import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """This Class encapsulates the behavior of a multi-layer perceptron
    or a simple fully connected neural network.
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # reshapes 2D image to 1D tensor
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conducts MLP forward pass. Terminates in softmax layer to 
        convert activations to probabilities.
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNN(nn.Module):
    """Class that encapsulates a 2D convolutional network. Uses the LeNet architecture
    for the model hyperparameters.
    """

    def __init__(self,):
        super(CNN, self).__init__()
        self.unflatten = nn.Unflatten(1, (1, 28, 28))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) #((28 - 5 + 2(2)) / 1) + 1 = 28 x 28 x 6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) #((28 - 2)/2 + 1) = 14 x 14 x 6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,) #((14 - 5 + 2(0))/ 1) + 1 = 10 x 10 x 16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) #((10 - 2)/2 + 1) = 5 x 5 x 16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the CNN forward pass."""
        x = self.flatten(x) # flattening and unflattening makes deployment in main.py consistent between MLP and CNN models.
        x = self.unflatten(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
