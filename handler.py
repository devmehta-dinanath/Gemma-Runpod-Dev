import os
import time
import subprocess
import requests
import runpod


MODEL_PATH = os.getenv(
    "MODEL_NAME",
    "/runpod-volume/models/gemma-4-e2b"
)

HOST = "127.0.0.1"
PORT = 8000

MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.85")
MAX_NUM_SEQS = os.getenv("MAX_NUM_SEQS", "1")


vllm_process = None


def start_vllm():
    global vllm_process

    if vllm_process is not None:
        return

    command = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--limit-mm-per-prompt",
        '{"image":4,"audio":1,"video":1}',
    ]

    print("Starting vLLM:")
    print(" ".join(command))

    vllm_process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        env=os.environ.copy(),
    )

    print("vLLM process started.")


def wait_for_vllm():
    print("Waiting for vLLM server...")

    url = f"http://{HOST}:{PORT}/health"

    for attempt in range(180):
        try:
            response = requests.get(
                url,
                timeout=5
            )

            if response.status_code == 200:
                print("vLLM server is ready.")
                return

        except Exception:
            pass

        if vllm_process is not None:
            return_code = vllm_process.poll()

            if return_code is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {return_code}"
                )

        if attempt % 10 == 0:
            print(
                f"Still waiting for vLLM... "
                f"attempt {attempt + 1}/180"
            )

        time.sleep(2)

    raise TimeoutError(
        "vLLM server did not become ready within 6 minutes."
    )


def handler(job):
    try:
        job_input = job.get("input", job)

        if not isinstance(job_input, dict):
            return {
                "error": "Input must be a JSON object."
            }

        messages = job_input.get("messages")

        if not messages:
            return {
                "error": (
                    "Missing 'messages'. "
                    "Expected OpenAI-compatible chat messages."
                )
            }

        payload = {
            "model": MODEL_PATH,
            "messages": messages,
        }

        # Forward optional OpenAI/vLLM parameters
        optional_parameters = [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "stream",
            "seed",
        ]

        for parameter in optional_parameters:
            if parameter in job_input:
                payload[parameter] = job_input[parameter]

        print("Sending request to vLLM...")

        response = requests.post(
            f"http://{HOST}:{PORT}/v1/chat/completions",
            json=payload,
            timeout=600,
        )

        if response.status_code != 200:
            return {
                "error": "vLLM request failed",
                "status_code": response.status_code,
                "details": response.text,
            }

        result = response.json()

        print("vLLM request completed.")

        return result

    except Exception as e:
        print(f"Handler error: {type(e).__name__}: {e}")

        return {
            "error": str(e),
            "type": type(e).__name__,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Gemma 4 E2B RunPod Serverless Worker")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Max model length: {MAX_MODEL_LEN}")
    print(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"Max sequences: {MAX_NUM_SEQS}")
    print("=" * 60)

    start_vllm()
    wait_for_vllm()

    print("Starting RunPod Serverless handler...")

    runpod.serverless.start({
        "handler": handler
    })
