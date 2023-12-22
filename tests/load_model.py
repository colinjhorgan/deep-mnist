import os

import torch

from src.mlp import MLP, test_mlp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = MLP()
model.load_state_dict(torch.load(os.path.join(BASE_DIR, '../models/mnist-mlp.pt')))
model.eval()

data = torch.randn((64, 784))

t1 = torch.tensor([1, 2, 3, 4, 5])
t2 = torch.tensor([1, 3, 2, 4, 5])

test_mlp(model)
