"""L200 — ViT on CIFAR-10, DDP across 8 TPU chips, Ray Data ingest.

Port of rayexamples/distributing-pytorch (step 3) to TPU. Shows the full
loop a real training job needs: Ray Data shards → DDP forward/backward →
ray.train.report + checkpoint.
"""

import os
import tempfile
import uuid

import torch
import torch.distributed as dist
from torch import nn
from torchvision.models import VisionTransformer
from torchvision.transforms import Compose, Normalize, ToTensor

import ray
import ray.data
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def transform_cifar(row):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    row["image"] = transform(row["image"])
    return row


def train_func_per_worker(config):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # torchvision's ViT uses nn.MultiheadAttention; its eval-mode fastpath calls
    # aten::_native_multi_head_attention which torch_tpu hasn't implemented.
    # (torch.backends is a GenericModule wrapper that breaks cloudpickle, so
    # reach the real submodule via importlib instead of attribute access.)
    import importlib

    importlib.import_module("torch.backends.mha").set_fastpath_enabled(False)

    train_shard = ray.train.get_dataset_shard("train")
    valid_shard = ray.train.get_dataset_shard("valid")

    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=8,
        hidden_dim=384,
        mlp_dim=768,
        num_classes=10,
    )

    ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(epochs):
        model.train()
        for batch in train_shard.iter_torch_batches(
            batch_size=batch_size, device="tpu"
        ):
            X, y = batch["image"], batch["label"]
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss, num_correct, num_total, num_batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in valid_shard.iter_torch_batches(
                batch_size=batch_size, device="tpu"
            ):
                X, y = batch["image"], batch["label"]
                pred = model(X)
                loss = loss_fn(pred, y)

                valid_loss += loss.item()
                num_total += y.shape[0]
                num_batches += 1
                num_correct += (pred.argmax(1) == y).sum().item()

        valid_loss /= max(num_batches, 1)
        accuracy = num_correct / max(num_total, 1)

        # torch_tpu's deferred execution means every rank must hit the same
        # materialization points; rank-0-only checkpointing would hang the rest.
        # And torch.save's storage.cpu() path is broken on torch_tpu (reports
        # byte count where element count is expected) — pre-move each tensor.
        dist.barrier()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as ckpt_dir:
            torch.save(state, os.path.join(ckpt_dir, "model.pt"))
            ray.train.report(
                metrics={"loss": valid_loss, "accuracy": accuracy},
                checkpoint=ray.train.Checkpoint.from_directory(ckpt_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print({"epoch": epoch, "loss": valid_loss, "accuracy": accuracy})


if __name__ == "__main__":
    storage_path = os.environ.get("RAY_STORAGE_PATH")

    train_ds = ray.data.read_parquet("s3://ray-example-data/cifar10-parquet/train").map(
        transform_cifar
    )
    valid_ds = ray.data.read_parquet("s3://ray-example-data/cifar10-parquet/test").map(
        transform_cifar
    )

    num_workers = int(os.environ.get("NUM_TPU_WORKERS", "8"))
    global_batch_size = 512
    train_config = {
        "lr": 1e-3,
        "epochs": 1,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    scaling_config = ScalingConfig(
        use_tpu=True,
        num_workers=num_workers,
        topology="2x2x1",
        accelerator_type="TPU-V7X",
        placement_strategy="PACK",
        resources_per_worker={"TPU": 0.5},
    )
    run_config = RunConfig(
        storage_path=storage_path,
        name=f"vit-cifar10-tpu-{uuid.uuid4().hex[:8]}",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        datasets={"train": train_ds, "valid": valid_ds},
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training result: {result.metrics}")
    print(f"Best checkpoint: {result.checkpoint}")
