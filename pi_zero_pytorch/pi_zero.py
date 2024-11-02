import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

from autoregressive_diffusion_pytorch import (
    AutoregressiveFlow
)

from x_transformers import (
    Decoder
)

class PiZero(Module):
    def __init__(self):
        super().__init__()

    def forward(self, state):
        return actions
