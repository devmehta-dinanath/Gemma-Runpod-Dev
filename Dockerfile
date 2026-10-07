FROM runpod/worker-v1-vllm:v2.27.1

RUN pip install --no-cache-dir \
    "vllm[audio]==0.29.0" \
    av \
    librosa \
    soundfile
