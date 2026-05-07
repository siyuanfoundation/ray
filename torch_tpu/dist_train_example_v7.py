# to test the script
# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python dist_train_example.py
#
import os

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import SGD

from torch_tpu import api

import ray
import ray.data
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


def train_func():
    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]
    tpu_device = api.tpu_device()
    print(
        f"Rank {rank}: Device type: {tpu_device.type}, Device index:"
        f" {tpu_device.index if tpu_device.index is not None else 'default'}"
    )
    # print all TPU related environment variables
    print(f"TPU related environment variables on Rank {rank}/{world_size}:")
    for env_var in os.environ:
        if "TPU" in env_var or "XLA" in env_var:
            print(f"{env_var}: {os.environ[env_var]}")

    # Simple linear regression: y = 2x + 1
    model = nn.Linear(1, 1)

    # Prepare model for distributed training
    model.to("tpu")

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Get the dataset shard for this worker
    train_shard = train.get_dataset_shard("train")

    for epoch in range(10):
        # Iterate over the shard using iter_torch_batches
        for batch in train_shard.iter_torch_batches(batch_size=32):
            # batch is a dict of tensors: {"x": tensor, "y": tensor}
            inputs = batch["x"].float().unsqueeze(1).to("tpu")
            targets = batch["y"].float().unsqueeze(1).to("tpu")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Report metrics to Ray Train
        train.report({"loss": loss.item()})
        if rank == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")


if __name__ == "__main__":
    # Create a simple dataset using Ray Data
    x = np.random.randn(100).astype(np.float32)
    y = 2 * x + 1 + 0.1 * np.random.randn(100).astype(np.float32)
    df = pd.DataFrame({"x": x, "y": y})
    dataset = ray.data.from_pandas(df)

    # Scaling configuration for CPU training
    scaling_config = ScalingConfig(
        use_tpu=True,
        num_workers=8,
        topology="2x2x1",
        accelerator_type="TPU-V7X",
        placement_strategy="PACK",
        resources_per_worker={"TPU": 0.5},
    )

    # Initialize TorchTrainer with the dataset
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        datasets={"train": dataset},
    )

    # Run the trainer
    result = trainer.fit()
    print(f"Training completed. Result: {result}")
