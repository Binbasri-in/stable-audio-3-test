import logging
import os
import typing as tp
import warnings
from enum import Enum

import torch
from einops import rearrange
from torch import nn

from .blocks import ExpoFourierFeatures, FourierFeatures
from .utils import enable_torch_compile


class PaddingMode(str, Enum):
    NONE = "none"
    ZERO = "zero"
    LEARNED = "learned"


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False, padding_mode: str = "zero"):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings),
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any, device: tp.Any) -> tp.Any:
        raise NotImplementedError()


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256, fourier_features_type: tp.Literal["learned", "expo"] = "learned"):
        super().__init__()
        self.features = features
        if fourier_features_type == "expo":
            self.embedding = nn.Sequential(ExpoFourierFeatures(dim=dim), nn.Linear(in_features=dim, out_features=features))
        else:
            self.embedding = nn.Sequential(FourierFeatures(in_features=1, out_features=dim), nn.Linear(in_features=dim, out_features=features))

    def forward(self, x: tp.Union[tp.List[float], torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x


class NumberConditioner(Conditioner):
    """Conditioner that takes a list of floats, normalizes them for a given range, and returns embeddings."""

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1, fourier_features_type: tp.Literal["learned", "expo"] = "learned"):
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
        self.embedder.to(device)
        floats = [float(x) for x in floats]
        floats_t = torch.tensor(floats, device=device).clamp(self.min_val, self.max_val)
        normalized_floats = (floats_t - self.min_val) / (self.max_val - self.min_val)

        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)
        return [float_embeds, torch.ones(float_embeds.shape[0], 1, device=device)]


class T5GemmaConditioner(Conditioner):
    T5GEMMA_MODELS = ["google/t5gemma-b-b-ul2"]
    T5GEMMA_MODEL_DIMS = {"google/t5gemma-b-b-ul2": 768}

    def __init__(
        self,
        output_dim: int,
        model_name: str = "google/t5gemma-b-b-ul2",
        max_length: int = 128,
        enable_grad: bool = False,
        project_out: bool = False,
        padding_mode: str = "zero",
        model_path: str = None,
        repo_id: str = None,
        subfolder: str = None,
    ):
        assert model_name in self.T5GEMMA_MODELS, f"Unknown T5 model name: {model_name}"
        super().__init__(self.T5GEMMA_MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        load_from = model_path or repo_id or model_name
        self.max_length = max_length
        self.enable_grad = enable_grad

        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel

                hf_kwargs = {"subfolder": subfolder} if subfolder else {}
                self.tokenizer = AutoTokenizer.from_pretrained(load_from, **hf_kwargs)
                config = AutoConfig.from_pretrained(load_from, **hf_kwargs)
                config.is_encoder_decoder = False
                model = (
                    T5GemmaEncoderModel.from_pretrained(load_from, config=config, **hf_kwargs)
                    .train(enable_grad)
                    .requires_grad_(enable_grad)
                )
            finally:
                logging.disable(previous_level)
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if enable_torch_compile:
            model = torch.compile(model)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self._device_initialized = False

    def forward(
        self,
        inputs: tp.Union[tp.List[str], tp.List[tp.Dict[str, torch.Tensor]]],
        device: tp.Union[torch.device, str],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if not self._device_initialized:
            self.model.to(device)
            self.proj_out.to(device)
            self.model.eval()
            self._device_initialized = True

        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)
        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Dict[str, str] = None,
        pre_encoded_keys: tp.List[str] = None,
    ):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}
        self.pre_encoded_keys = pre_encoded_keys or []

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key
            conditioner_inputs = []

            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                val = x[condition_key]
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    conditioner_input = val[0]
                else:
                    conditioner_input = val
                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any], pretransform=None
) -> MultiConditioner:
    conditioners = {}
    cond_dim = config["cond_dim"]
    default_keys = config.get("default_keys", {})
    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        cid = conditioner_info["id"]
        ctype = conditioner_info["type"]
        cconfig = {"output_dim": cond_dim}
        cconfig.update(conditioner_info["config"])

        if ctype == "t5gemma":
            conditioners[cid] = T5GemmaConditioner(**cconfig)
        elif ctype == "number":
            conditioners[cid] = NumberConditioner(**cconfig)
        else:
            raise ValueError(f"Unsupported conditioner type for Stable Audio 3: {ctype}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)
