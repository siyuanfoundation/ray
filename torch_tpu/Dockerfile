# Use the latest Ray master as base.
FROM rayproject/ray:nightly.260501.1c7c90-py312-tpu
# Invalidate the cache so that fresh code is pulled in the next step.
ARG BUILD_DATE
# Retrieve your development code from the parent directory.
ADD . ray
# Install symlinks to your modified Python code.
RUN python ray/python/ray/setup-dev.py --skip=serve -y

RUN sudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
RUN pip uninstall -y torch torchvision scipy

RUN pip install keyrings.google-artifactregistry-auth
RUN pip install pandas==2.3.3

ARG AUTH_TOKEN
# Set the PIP_INDEX_URL environment variable to point to the Torch TPU virtual registry
ENV PIP_INDEX_URL="https://oauth2accesstoken:${AUTH_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

ARG GITHUB_USER
ARG GITHUB_TOKEN
RUN git clone --depth 1 --branch v0.19.0 https://github.com/vllm-project/vllm.git
RUN git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/google-pytorch/torchtpu-vllm.git
# Patch vLLM's TPU requirements to accept our local workspace in torchtpu-vllm instead of overriding it
RUN sed -i '/tpu-inference/d' vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.17.1 base version to prevent .dev prerelease mismatch during dependency resolution)
WORKDIR torchtpu-vllm
RUN pip install --upgrade pip && SETUPTOOLS_SCM_PRETEND_VERSION=0.19.0 VLLM_TARGET_DEVICE="tpu" pip install -e ../vllm
RUN pip install --pre -e .

# Freeze torch_tpu version until head is fixed
RUN pip install --pre torchvision torch_tpu==0.1.1.dev20260423092951
