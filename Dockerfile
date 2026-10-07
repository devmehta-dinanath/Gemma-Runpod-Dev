FROM vllm/vllm-openai:v0.29.0-cu129

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/:/worker/src"
ENV HF_HOME="/runpod-volume/huggingface-cache"
ENV HF_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TOKENIZERS_PARALLELISM=false

# Install git so we can bring in the matching RunPod worker code
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Get the RunPod vLLM worker code matching v2.27.0
RUN git clone --depth 1 --branch v2.27.0 \
    https://github.com/runpod-workers/worker-vllm.git \
    /worker

# Install RunPod worker dependencies
RUN pip install --no-cache-dir \
    runpod==1.9.1 \
    ray \
    pandas \
    pyarrow \
    huggingface-hub \
    packaging \
    typing-extensions \
    pydantic \
    pydantic-settings \
    hf-transfer \
    bitsandbytes

# Audio support
RUN pip install --no-cache-dir \
    "av" \
    "librosa" \
    "soundfile"

RUN chmod +x /worker/src/start.sh

CMD ["/bin/bash", "/worker/src/start.sh"]
