import json
from huggingface_hub import hf_hub_download

from .factory import create_model_from_config
from .models.utils import load_ckpt_state_dict


def get_pretrained_model(name: str):
    """Download model config and checkpoint from Hugging Face Hub and load model."""
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type="model")

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type="model")
    except Exception:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type="model")

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))
    return model, model_config
