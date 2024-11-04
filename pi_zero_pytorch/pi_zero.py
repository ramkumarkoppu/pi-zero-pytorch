from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor
from torch.nn import Module, ModuleList
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum, pack, unpack

# constants

flex_attention = torch.compile(flex_attention)

LinearNoBias = partial(nn.Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_actions_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_actions_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        multimodal_seq,
        actions
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        mq, mk, mv, aq, ak, av = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av))

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, v)))

        # attention

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        causal_mask = torch.ones(sim.shape[-2], dtype = torch.bool, device = device).triu(1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        return self.to_out(mout), self.to_actions_out(aout)

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
