import os

import torch
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL  # type: ignore
from sgm.models.autoencoder import AutoencoderKL as SgmAutoencoderKL
from torch import nn


def _resolve_local_vae_path() -> str | None:
    env_vae_path = os.environ.get("SEVA_VAE_PATH")
    if env_vae_path:
        env_vae_path = os.path.expanduser(env_vae_path)
        if os.path.isdir(env_vae_path):
            env_vae_path = os.path.join(env_vae_path, "sdxl_vae.safetensors")
        if os.path.isfile(env_vae_path):
            return env_vae_path

    repo_vae_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sdxl-vae",
        "sdxl_vae.safetensors",
    )
    if os.path.isfile(repo_vae_path):
        return repo_vae_path

    return None


def _build_local_sdxl_vae(ckpt_path: str) -> SgmAutoencoderKL:
    return SgmAutoencoderKL(
        ckpt_path=ckpt_path,
        embed_dim=4,
        monitor="val/rec_loss",
        ddconfig={
            "attn_type": "vanilla-xformers",
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        lossconfig={"target": "torch.nn.Identity"},
    )


class AutoEncoder(nn.Module):
    scale_factor: float = 0.18215
    downsample: int = 8

    def __init__(self, chunk_size: int | None = None):
        super().__init__()
        local_vae_path = _resolve_local_vae_path()
        self.uses_sgm_autoencoder = local_vae_path is not None
        if local_vae_path is not None:
            self.module = _build_local_sdxl_vae(local_vae_path)
        else:
            self.module = DiffusersAutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                subfolder="vae",
                force_download=False,
                low_cpu_mem_usage=False,
            )
        self.module.eval().requires_grad_(False)  # type: ignore
        self.chunk_size = chunk_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_sgm_autoencoder:
            return self.module.encode(x)  # type: ignore
        return (
            self.module.encode(x).latent_dist.mean  # type: ignore
            * self.scale_factor
        )

    def encode(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._encode(x_chunk) for x_chunk in x.split(chunk_size)],
                dim=0,
            )
        else:
            return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.uses_sgm_autoencoder:
            return self.module.decode(z)  # type: ignore
        return self.module.decode(z / self.scale_factor).sample  # type: ignore

    def decode(self, z: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._decode(z_chunk) for z_chunk in z.split(chunk_size)],
                dim=0,
            )
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
