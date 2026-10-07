FROM vllm/vllm-openai:v0.29.0-cu129

ENV DEBIAN_FRONTEND=noninteractive

ENV HF_HOME=/runpod-volume/huggingface-cache
ENV HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub
ENV TOKENIZERS_PARALLELISM=false

RUN pip install --no-cache-dir \
    av==19.0.1 \
    librosa==1.0.0 \
    soundfile==0.14.0 \
    runpod \
    requests

COPY handler.py /handler.py

ENTRYPOINT ["python", "-u"]
CMD ["/handler.py"]
