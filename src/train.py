import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2 
from torchvision import datasets
from torch.utils.data import DataLoader

from helpers import get_device
from models import MLP, CNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model(
        model,
        optimizer: optim.Optimizer = optim.SGD,
        lr: float = 0.01,
        epochs: int = 20):
    """Executes training loop for a given model, given an optizer with a learning
    rate, and a number of epochs.

    *** Note that data is expected to be obtained by /config.py prior to executing
    any training. ***
    """
    batch_size = 200
    device = get_device()
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,),(0.3081,)), # mean and std of training data
        v2.RandomRotation(degrees=30),
        v2.RandomResizedCrop(size=(28,28), scale=(0.6, 1), antialias=True)
    ])
    train_dataset = datasets.MNIST(
        os.path.join(BASE_DIR, '../data'),
        train=True,
        download=True,
        transform=transform,)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4, 
    )
    print(f"Training Device: {device}")
    print(f"Size of training dataset: {len(train_dataset)}")
    print(f"Number of batches: {batch_size}")
    print(f"Batches per epoch: {len(train_dataset)//(batch_size)}")
    
    model.to(device)
    model.train()
    objective_f = nn.CrossEntropyLoss()
    optimizer=optimizer(model.parameters(), lr=lr)
    
    with tqdm(total=epochs) as prog_bar:
        for epoch in range(epochs):
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = objective_f(outputs, label)
                loss.backward()
                optimizer.step()
            prog_bar.update(1)
            prog_bar.set_description(desc=f"**Training Loss: {loss.item():.4f}")

    return model


def test_model(
        model,
    ):
    """ Executes model testing."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,),(0.3081,)), # mean and std of training data
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='mlp',
        help='Model architecture to use for training. Must be one of MLP or '
        'CNN, case insensitive',
    )
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=10,
        help='Number of epochs of model training to conduct',
    )
    parser.add_argument(
        '--filename',
        '-f',
        type=str,
        help='Name of file to save model binary as in the `models` directory.'
        'Should end in .pt by convention.'
    )
    args = parser.parse_args()

    if args.model.upper() not in ['MLP','CNN']:
        raise ValueError("--model must be one of 'MLP', or 'CNN'")

    model = MLP() if args.model.upper() == 'MLP' else CNN()
    epochs = args.epochs

    torch.manual_seed(0) # set manual seed for reproducibility*
    print(f"Model Type: {args.model.upper()}")
    final_model = train_model(model=model, epochs=epochs)
    test_model(model)
    torch.save(model.state_dict(), os.path.join(BASE_DIR, f'../models/{args.filename}'))
