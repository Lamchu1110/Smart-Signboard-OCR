from pathlib import Path

import torch


class VietOCRRecognizer:
    def __init__(self, config_path: Path, weights_path: Path, device: str):
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        self.device = device

        if not self.config_path.exists():
            raise FileNotFoundError(f"VietOCR config not found: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"VietOCR weights not found: {self.weights_path}")

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise ImportError(
                "VietOCR is required for OCR_ENGINE=vietocr. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        config = Cfg.load_config_from_file(str(self.config_path))
        config["weights"] = str(self.weights_path)
        config["device"] = self.device
        config["predictor"]["beamsearch"] = False

        # Avoid downloading the original pretrained backbone during inference.
        if "cnn" in config and isinstance(config["cnn"], dict):
            config["cnn"]["pretrained"] = False

        self.predictor = Predictor(config)

    @classmethod
    def from_paths(cls, config_path: Path, weights_path: Path):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return cls(config_path=config_path, weights_path=weights_path, device=device)

    def recognize(self, image):
        return self.predictor.predict(image.convert("RGB"))
