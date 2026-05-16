# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu --runtime-env-json='{"env_vars": {"RAY_STORAGE_PATH": "<your_storage_path>"}}' -- python torch_trainer_example.py

import os
import tempfile

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import ray.train
from ray.train.torch import TorchTrainer


# Define your network structure.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))


# Training loop.
def train_fn_per_worker(config):

    # Read configurations.
    lr = config["lr"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    # Fetch training dataset.
    train_dataset_shard = ray.train.get_dataset_shard("train")

    # Instantiate and prepare model for training.
    model = NeuralNetwork()
    model = ray.train.torch.prepare_model(model)

    # Define loss and optimizer.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create data loader.
    dataloader = train_dataset_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float
    )

    # Train multiple epochs.
    for epoch in range(num_epochs):
        # Train epoch.
        for batch in dataloader:
            output = model(batch["input"])
            loss = loss_fn(output, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create checkpoint.
        base_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # Move state dict to CPU before saving to avoid torch_tpu size mismatch bug
            # where storage.cpu() reports byte count (128) instead of element count (32).
            state_dict = {
                k: v.detach().cpu() for k, v in base_model.state_dict().items()
            }
            torch.save(
                {"model_state_dict": state_dict},
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            # Report metrics and checkpoint.
            ray.train.report({"loss": loss.item()}, checkpoint=checkpoint)


# Define datasets.
train_dataset = ray.data.from_items(
    [{"input": [x], "label": [2 * x + 1]} for x in range(128)]
)

scaling_config = ray.train.ScalingConfig(
    use_tpu=True,
    num_workers=8,
    topology="2x4",
    accelerator_type="TPU-V6E",
    placement_strategy="PACK",
    resources_per_worker={
        "TPU": 1
    },  # use 1 TPU chips per worker for v6, 0.5 TPU for v7 to account for the 2 cores per TPU chip
)

# Initialize the Trainer.
storage_path = os.environ.get("RAY_STORAGE_PATH")
trainer = TorchTrainer(
    train_fn_per_worker,
    train_loop_config={"num_epochs": 1, "lr": 0.01, "batch_size": 32},
    scaling_config=scaling_config,
    run_config=ray.train.RunConfig(storage_path=storage_path),
    datasets={"train": train_dataset},
)

# Train the model.
result = trainer.fit()

# Inspect the results.
final_loss = result.metrics["loss"]
