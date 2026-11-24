# CUDA 13 runtime + the few libs Unsloth's prebuilt llama-server links against.
# The binaries themselves are bind-mounted from ./opt/llama-unsloth-b12588 (see the README).
FROM nvidia/cuda:13.4.3-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
