FROM runpod/worker-v1-vllm:v2.19.0

RUN pip install --no-cache-dir \
    av \
    librosa \
    soundfile
