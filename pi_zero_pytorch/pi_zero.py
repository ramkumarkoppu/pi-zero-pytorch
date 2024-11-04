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

        self.rmsnorm = nn.RMSNorm(dim)
        self.actions_rmsnorm = nn.RMSNorm(dim)

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

        multimodal_seq = self.rmsnorm(multimodal_seq)
        actions = self.actions_rmsnorm(actions)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        mq, mk, mv, aq, ak, av = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av))

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

        # attention

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        return self.to_out(mout), self.to_actions_out(aout)

# attention

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dim_inner = None
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim)
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        seq
    ):
        seq = self.rmsnorm(seq)
        seq, gates = self.proj_in(seq).chunk(2, dim = -1)
        seq = seq * F.gelu(gates)
        return self.proj_out(seq)

# main class

class PiZero(Module):
    def __init__(
        self,
        dim,
        num_tokens,
        dim_action_input,
        depth = 12,
        dim_head = 64,
        heads = 8,
        ff_expand_factor = 4.,
        vit: Module | None = None
    ):
        super().__init__()

        self.vit = vit

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = []

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor)
            ]))

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim)
        self.final_actions_norm = nn.RMSNorm(dim)

        self.state_to_logits = LinearNoBias(dim, num_tokens)
        self.actions_to_denoised_pred = LinearNoBias(dim, dim_action_input)

    def forward(
        self,
        images,    # vision
        token_ids, # language
        actions    # action
    ):

        tokens = self.token_emb(token_ids)

        if exists(self.vit):
            assert images.ndim == 4

            with torch.no_grad():
                self.vit.eval()
                visual_tokens = self.vit(images)
        else:
            assert images.ndim == 3, 'images must be already encoded'
            visual_tokens = images

        # concat visual rep with language

        state, packed_shape = pack([visual_tokens, tokens], 'b * d')

        # transformer

        for attn, state_ff, actions_ff in self.layers:

            state_out, actions_out = attn(state, actions)

            state = state + state_out
            actions = actions + actions_out

            state = state_ff(state) + state
            actions = actions_ff(actions) + actions

        # unpack and unembed to predictions

        visual_tokens, tokens = unpack(state, packed_shape, 'b * d')

        state = self.final_norm(state)
        actions = self.final_actions_norm(actions)

        state_logits = self.state_to_logits(state)
        denoised_actions = self.actions_to_denoised_pred(actions)

        return state_logits, denoised_actions

# fun

π0 = PiZero
