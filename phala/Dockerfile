# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm@sha256:f040863673aea2570c3ff6a5c3fb4c673a016cbc5375005ad145915922b6b78a

ARG SOURCE_REVISION
ARG VERSION
LABEL org.opencontainers.image.title="Kev-4B decision service" \
      org.opencontainers.image.source="https://github.com/Phala-Network/kev" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      ai.phala.kev.source="https://github.com/jaredpalmer/kev" \
      ai.phala.kev.upstream-revision="5e42a7a03f28134853dd3ff77461457e921e5ec1"

ENV PATH=/opt/kev/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    KEV_BACKEND=torch \
    KEV_DTYPE=bf16 \
    KEV_FUSED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*
RUN test -n "${SOURCE_REVISION}" && test -n "${VERSION}"
COPY phala/bootstrap-requirements.txt /opt/build/bootstrap-requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /opt/build/bootstrap-requirements.txt
WORKDIR /opt/kev
# Build context must be a clean archive of the exact fork source release.
# The runtime and frozen lock are taken from that archive, never fetched from upstream.
COPY . /opt/kev
RUN uv sync --frozen --no-dev --extra serve --no-editable --python /usr/local/bin/python

# The author's CUDA recipe explicitly overrides torch 2.8's Triton 3.4 pin:
# FLA's Hopper kernels require >=3.7.1. Fix the selected override, never resolve
# it dynamically at service startup. causal-conv1d matches cp313/cu12/torch2.8.
COPY phala/cuda-requirements.txt /opt/build/cuda-requirements.txt
RUN uv pip install --python /opt/kev/.venv/bin/python --no-deps \
      --require-hashes -r /opt/build/cuda-requirements.txt
COPY phala/verify_image.py /opt/kev-verify/verify_image.py
RUN hf --help >/dev/null && hf download --help >/dev/null \
    && python /opt/kev-verify/verify_image.py

EXPOSE 8008
CMD ["python", "-m", "kev.serve", "--host", "0.0.0.0", "--port", "8008", "--run", "/models/kev"]
