# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python single_tpu_train_example.py

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch_tpu import api

import ray


@ray.remote(resources={"TPU": 1})
def train_linear_regression():
    print("TPU related environment variables:")
    for env in os.environ:
        if "TPU" in env or "XLA" in env:
            print(f"{env}: {os.environ[env]}")
    # Initialize the TPU device
    device = api.tpu_device()
    if device is None:
        print("No TPU device found in the remote task.")
        return

    print(f"Running on TPU device: {device}")

    # Dummy data: y = 2*x1 + 3*x2 + 1
    torch.manual_seed(42)
    x_train = torch.randn(100, 2, device=device)
    y_target = (
        2 * x_train[:, 0:1]
        + 3 * x_train[:, 1:2]
        + 1
        + 0.1 * torch.randn(100, 1, device=device)
    )

    # Model with 2 input features
    model = nn.Linear(2, 1).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    final_loss = -1.0

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        final_loss = loss.item()

    # Final parameters
    for name, param in model.named_parameters():
        print(f"Final Parameters: {name}: {param.data}")
    return final_loss


if __name__ == "__main__":
    ray.init()
    start_time = time.time()
    task = train_linear_regression.remote()
    final_loss = ray.get(task)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(f"Final loss: {final_loss}")
