import os
import torch
from safetensors.torch import load_file

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function
    return function


def load_ckpt_state_dict(ckpt_path: str):
    """Load model weights from .safetensors or .ckpt / .pt torch checkpoint."""
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    return state_dict


def copy_state_dict(model: torch.nn.Module, state_dict: dict):
    """Load state_dict into model for matching keys and shapes."""
    model_state_dict = model.state_dict()
    for key, value in state_dict.items():
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            if isinstance(value, torch.nn.Parameter):
                value = value.data
            model_state_dict[key] = value
        else:
            print(f"[copy_state_dict] Key {key} not found or shape mismatch. Skipping.")
    model.load_state_dict(model_state_dict, strict=False)
