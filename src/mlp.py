import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_device(device: Optional[str] = None) -> torch.device:
    """Determines the device to send model parameters to for training.
    Prioritizes platforms that have GPU first. If none are found, resorts
    to using CPU.

    Parameters
    ----------
    device: str
        Desired device to use (if any). Must be a recognizeable device by
        torch.device
    """
    if device:
        return torch.device(device)
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return torch.device(device)


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


def train_mlp(
        model: MLP = MLP(),
        optimizer: optim.Optimizer = optim.SGD,
        lr: float = 0.01,
        epochs: int = 20):
    """Executes training loop for MLP class given an optizer with a learning
    rate, and a number of epochs.

    *** Note that data is expected to be obtained by /config.py prior to executing
    any training. ***
    """
    batch_size = 200
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081)) # mean and std of training data
    ])
    train_dataset = datasets.MNIST(
        os.path.join(BASE_DIR, '../data'),
        train=True,
        download=True,
        transform=transform,)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = get_device()
    print(f"Training Device: {device}")
    print(f"Size of training dataset: {len(train_dataset)}")
    print(f"Number of batches: {batch_size}")
    print(f"Batches per epoch: {len(train_dataset)//(batch_size)}")
    
    model.to(device)
    model.train()
    objective_f = nn.CrossEntropyLoss()
    optimizer=optimizer(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = objective_f(outputs, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item()}")
    return model


def test_mlp(
        model: MLP,
    ):
    """ Executes model testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081)) # mean and std of training data
    ])
    test_dataset = datasets.MNIST(
        os.path.join(BASE_DIR, '../data'),
        train=False,
        download=True,
        transform=transform,)
    test_loader = DataLoader(test_dataset, batch_size=200)
    print(f"Size of testing dataset: {len(test_dataset)}")
    
    device = get_device()
    model.to(device)
    objective_f = nn.CrossEntropyLoss()

    with torch.no_grad():
        loss = 0 
        correct = 0
        model.eval()

        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            correct += labels.eq(predictions).sum()
            loss += objective_f(outputs, labels)

        accuracy = correct / len(test_dataset)

    print(f"Final Loss: {loss}")
    print(f"Final Accuracy: {accuracy}")

if __name__ == "__main__":
    model = train_mlp(epochs=20)
    test_mlp(model)
    torch.save(model.state_dict(), os.path.join(BASE_DIR, '../models/mnist-mlp.pt'))
