# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python batch_inference_llm.py

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen3-0.6B",
    engine_kwargs={
        "tensor_parallel_size": 1,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
    },
    # use TPU instead of GPU
    placement_group_config={
        "bundles": [{"TPU": 1, "CPU": 1, "GPU": 0}],
    },
    concurrency=1,
    batch_size=1,
)
processor = build_processor(
    config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system", "content": "You are a bot that responds with haikus."},
            {"role": "user", "content": row["item"]},
        ],
        sampling_params=dict(
            temperature=0,
            max_tokens=250,
        ),
    ),
    postprocess=lambda row: dict(answer=row["generated_text"]),
)

ds = ray.data.from_items(["Start of the haiku is: Complete this for me..."])

ds = processor(ds)

print("=== Predictions ===")
ds.show(limit=1)

ray.shutdown()
