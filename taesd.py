"""
Tiny AutoEncoder for Stable Diffusion (TAEF2 decoder only)
Minimal extraction from https://github.com/madebyollin/taesd
Used for fast Flux 2 latent previews during generation.
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1 = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False)
            n_gn = n_in * 4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def Decoder(latent_channels=32, use_midblock_gn=True):
    """TAEF2 decoder: 32-channel latents -> RGB [0, 1]."""
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def convert_diffusers_keys(sd):
    """Convert diffusers-format safetensors keys to taesd.py layer indices."""
    out = {}
    for k, v in sd.items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 1 if encdec == "decoder" else 0
        out[".".join([str(int(index) + offset), *suffix])] = v
    return out


def load_taef2_decoder(safetensors_path):
    """Load the TAEF2 decoder from a safetensors file."""
    import safetensors.torch as stt
    decoder = Decoder(latent_channels=32, use_midblock_gn=True)
    sd = stt.load_file(safetensors_path)
    # Filter to decoder keys, apply +1 index offset, strip prefix
    decoder_weights = {}
    for k, v in sd.items():
        if not k.startswith("decoder."):
            continue
        _dec, _layers, index, *suffix = k.split(".")
        new_key = ".".join([str(int(index) + 1), *suffix])
        decoder_weights[new_key] = v
    decoder.load_state_dict(decoder_weights)
    return decoder.eval().requires_grad_(False)
