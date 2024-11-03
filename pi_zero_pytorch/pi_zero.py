import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

from transfusion_pytorch import (
    Transfusion
)

from x_transformers import (
    Decoder
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class PiZero(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        state
    ):
        return state
