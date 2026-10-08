FROM vllm/vllm-openai:v0.29.0-cu129

ENV DEBIAN_FRONTEND=noninteractive

ENV HF_HOME=/runpod-volume/huggingface-cache
ENV HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm-cache
ENV TORCHINDUCTOR_CACHE_DIR=/runpod-volume/vllm-cache/inductor

# This image ships python3 only. A start command of `python` exits 127.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python

# Audio uses soundfile/librosa/soxr. vLLM 0.29 removed the PyAV video
# decoder, so video is decoded with OpenCV.
RUN pip install --no-cache-dir \
    av==19.0.1 \
    librosa==1.0.0 \
    soundfile==0.14.0 \
    "soxr>=0.5.0" \
    "opencv-python-headless>=4.10" \
    runpod \
    requests

COPY handler.py /handler.py
RUN chmod 755 /handler.py

ENTRYPOINT ["/usr/bin/python3", "-u", "/handler.py"]
CMD []
