import os
import platform
import time

# Torch can hang on some Windows shells while platform.machine() queries WMI.
# Return the architecture directly before importing torch.
platform.machine = lambda: os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")

# On this Windows environment, importing ray before torch can make torch fail
# while loading c10.dll. Preload torch before ray/serve.
import torch  # noqa: F401

import ray
from ray import serve

from ocr import entrypoint


def main():
    os.environ.setdefault("OCR_ENGINE", "vietocr")

    if not ray.is_initialized():
        ray.init(address="auto", namespace="serve")

    serve.run(entrypoint)
    print("OCR API is running at http://localhost:8000")
    print("Press Ctrl+C to stop this process.")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping OCR API...")


if __name__ == "__main__":
    main()
