import json
from typing import Any, Dict

from .models.autoencoders import (
    AudioAutoencoder,
    create_decoder_from_config,
    create_encoder_from_config,
)
from .models.bottleneck import SoftNormBottleneck
from .models.diffusion import create_diffusion_cond_from_config
from .models.pretransforms import AutoencoderPretransform, PatchedPretransform


def create_model_from_config(model_config: Dict[str, Any]):
    model_type = model_config.get("model_type", None)
    assert model_type is not None, "model_type must be specified in model config"

    if model_type in ("diffusion_cond", "diffusion_cond_inpaint"):
        return create_diffusion_cond_from_config(model_config)
    elif model_type == "autoencoder":
        return create_autoencoder_from_config(model_config)
    else:
        raise NotImplementedError(f"Unsupported model type for Stable Audio 3: {model_type}")


def create_model_from_config_path(model_config_path: str):
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    return create_model_from_config(model_config)


def create_pretransform_from_config(pretransform_config: Dict[str, Any], sample_rate: int):
    pretransform_type = pretransform_config.get("type", None)
    assert pretransform_type is not None, "type must be specified in pretransform config"

    if pretransform_type == "autoencoder":
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = AutoencoderPretransform(
            autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked
        )
    elif pretransform_type == "patched":
        patched_config = pretransform_config["config"]
        pretransform = PatchedPretransform(**patched_config)
    else:
        raise NotImplementedError(f"Unsupported pretransform type for Stable Audio 3: {pretransform_type}")

    enable_grad = pretransform_config.get("enable_grad", False)
    pretransform.enable_grad = enable_grad
    pretransform.eval().requires_grad_(pretransform.enable_grad)
    return pretransform


def create_bottleneck_from_config(bottleneck_config: Dict[str, Any]):
    bottleneck_type = bottleneck_config.get("type", None)
    assert bottleneck_type is not None, "type must be specified in bottleneck config"

    if bottleneck_type == "softnorm":
        bottleneck = SoftNormBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f"Unsupported bottleneck type for Stable Audio 3: {bottleneck_type}")

    requires_grad = bottleneck_config.get("requires_grad", True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck


def create_autoencoder_from_config(config: Dict[str, Any]):
    ae_config = config["model"]
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in config"

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = None
    if "bottleneck" in ae_config:
        bottleneck = create_bottleneck_from_config(ae_config["bottleneck"])

    pretransform = None
    if "pretransform" in ae_config:
        pretransform = create_pretransform_from_config(ae_config["pretransform"], sample_rate)

    latent_dim = ae_config["latent_dim"]
    downsampling_ratio = ae_config["downsampling_ratio"]
    io_channels = ae_config["io_channels"]
    soft_clip = ae_config.get("decoder", {}).get("soft_clip", False)

    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        io_channels=io_channels,
        bottleneck=bottleneck,
        pretransform=pretransform,
        soft_clip=soft_clip,
    )
