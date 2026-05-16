# ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python serve_example.py

import os

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


def main():

    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id="my-model",
            model_source="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        ),
        accelerator_type="TPU-V6E",
        placement_group_config=dict(
            strategy="PACK",
            bundles=[{"TPU": 1}] * 8,
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=1,
                max_replicas=1,
            ),
            ############# Key Change ###############
            ray_actor_options={"num_gpus": 0},
        ),
        runtime_env=dict(env_vars={"HF_TOKEN": os.environ.get("HF_TOKEN")}),
        engine_kwargs={
            "tensor_parallel_size": 8,
            "max_model_len": 256,
            "max_num_batched_tokens": 256,
        },
    )

    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, blocking=True)


if __name__ == "__main__":
    main()
