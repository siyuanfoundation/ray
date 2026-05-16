# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python batch_inference_transformer_pipeline.py

from typing import Dict

import numpy as np

import ray

ds = ray.data.from_numpy(
    np.asarray(
        [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
            "The colors of the rainbow are",
        ]
    )
)


class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline

        # Set torch_tpu.api.tpu_device() as the device so the Huggingface pipeline uses TPU.
        # device = "cuda:0"
        from torch_tpu import api

        device = api.tpu_device()
        #########################
        self.model = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device=device)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["data"]), max_length=20, num_return_sequences=1
        )
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch


predictions = ds.map_batches(
    HuggingFacePredictor,
    # make sure num_gpus is set to 0, and resources is 1 TPU.
    num_gpus=0,
    resources={"TPU": 1},
    batch_size=2,
    # Set the concurrency up to the number of TPUs in your cluster.
    compute=ray.data.ActorPoolStrategy(size=2),
)

print("=== Predictions ===")
predictions.show(limit=4)

ray.shutdown()
