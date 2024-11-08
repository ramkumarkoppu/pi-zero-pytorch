from __future__ import annotations
from typing import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import pi, nn, tensor, is_tensor
from torch.nn import Module, ModuleList

from torchdiffeq import odeint

from rotary_embedding_torch import RotaryEmbedding

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum, pack, unpack

from pi_zero_pytorch.tensor_typing import Float, Int, Bool

import tqdm

# ein notation

# b - batch
# n - sequence
# na - seq of actions
# nt - seq of text tokens
# nv - seq of visual tokens
# d - dimension
# da - action dimension
# c - image channels
# h - image height
# w - image width
# f - image frames

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# flex attention related
# https://pytorch.org/blog/flexattention/

flex_attention = None

if torch.cuda.is_available():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention)

def create_pizero_attn_mask(prefix_causal_length):
    # the pi-zero attention is a triangular causal mask, but bidirectional attention for the actions at the very right hand side

    def inner(batch_index, head_index, query_index, key_index):
        return (
            query_index >= key_index and        # causal
            key_index >= prefix_causal_length   # bidirectional
        )

    return inner

def softclamp_score_mod(value):
    def identity(score, b, h, q, k):
        return score

    def softclamped(score, b, h, q, k):
        score = score / value
        score = torch.tanh(score)
        score = score * value
        return score

    return softclamped if value > 0. else identity

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    if value <= 0.:
        return t

    return (t / value).tanh() * value

# losses

def direction_loss(pred, target, dim = -1):
    return 0.5 * (1. - F.cosine_similarity(pred, target, dim = dim))

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softclamp_value = 50.,
        rotary_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = nn.RMSNorm(dim)

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_actions_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_actions_out = LinearNoBias(dim_inner, dim)

        self.softclamp_value = softclamp_value

    def forward_actions_with_cached_state(
        self,
        actions,
        cached_state_keys_values: tuple[Tensor, Tensor],
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: Callable | None = None
    ):
        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        aq, ak, av = tuple(self.split_heads(t) for t in (aq, ak, av))

        if exists(actions_value_residual):
            av = 0.5 * (av + actions_value_residual)

        q = aq
        mk, mv = cached_state_keys_values

        k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

        if exists(self.rotary_emb):
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # attention

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)
        else:
            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softclamp_value)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        actions_out = self.to_actions_out(out)

        if not return_keys_values:
            return actions_out

        return actions_out, (mk, mv, ak, av)

    def forward(
        self,
        multimodal_seq,
        actions,
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: Callable | None = None
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        multimodal_seq = self.rmsnorm(multimodal_seq)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        mq, mk, mv, aq, ak, av = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av))

        if exists(actions_value_residual):
            av = 0.5 * (av + actions_value_residual)

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)

        else:
            # attention

            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softclamp_value)

            causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

            causal_mask[..., seq_len:] = False  # actions have bidirectional attention, lining up with Transfusion paper

            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        output =  self.to_out(mout), self.to_actions_out(aout)

        if not return_keys_values:
            return output

        return output, (mk, mv, ak, av)

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

# actions need time conditioning
# ada-ln zero from DiT - here we will improvise with adaptive rmsnorm

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, '... -> ... 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    def forward(self, actions, cond):
        normed = self.norm(actions)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return normed * gamma + beta

class AdaptiveLayerscale(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(self, actions, cond):
        gamma = self.to_adaln_zero_gamma(cond)
        return actions * gamma.sigmoid()

# main class

class PiZero(Module):
    def __init__(
        self,
        dim,
        num_tokens,
        dim_action_input,
        dim_joint_state,
        dim_time_cond = None,
        depth = 12,
        dim_head = 64,
        heads = 8,
        use_flex_attn = False,
        ff_expand_factor = 4.,
        attn_softclamp_value = 50.,
        final_norm_softclamp_value = 30.,
        vit: Module | None = None,
        vit_dim = None,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        lm_loss_weight = 1.,
        flow_loss_weight = 1.,
        direction_loss_weight = 0.,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
    ):
        super().__init__()
        dim_time_cond = default(dim_time_cond, dim * 2)

        self.dim = dim

        # flex attention related

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention cannot be used'
        self.use_flex_attn = use_flex_attn
        self.attn_softclamp_value = attn_softclamp_value

        # vit

        self.vit = vit

        self.maybe_to_image_tokens = nn.Linear(vit_dim, dim) if vit_dim != dim else nn.Identity()

        # embedding

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.to_joint_state_tokens = nn.Linear(dim_joint_state, dim)

        self.dim_action_input = dim_action_input
        self.to_action_tokens = nn.Linear(dim_action_input, dim)

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            nn.Linear(dim, dim_time_cond),
            nn.SiLU(),
        )

        # positional embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # attention and feedforward

        layers = []
        cond_layers = []

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = self.rotary_emb, **attn_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs)
            ]))

            cond_layers.append(ModuleList([
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond),
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond)
            ]))

        self.layers = ModuleList(layers)
        self.cond_layers = ModuleList(cond_layers)

        self.final_norm_softclamp = partial(softclamp, value = final_norm_softclamp_value)

        self.final_norm = nn.RMSNorm(dim)
        self.final_actions_norm = nn.RMSNorm(dim)

        # unembedding

        self.state_to_logits = LinearNoBias(dim, num_tokens)
        self.actions_to_pred_flow = LinearNoBias(dim, dim_action_input)

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.flow_loss_weight = flow_loss_weight

        self.has_direction_loss = direction_loss_weight > 0.
        self.direction_loss_weight = direction_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        images,
        token_ids,
        joint_states,
        trajectory_length: int,
        reward_tokens = None,
        steps = 18,
        batch_size = 1,
        show_pbar = True
    ):
        was_training = self.training
        self.eval()

        pbar = tqdm.tqdm(desc = 'sampling action trajectory', disable = not show_pbar, total = steps)

        # ode step function

        cached_state_kv = None

        def ode_fn(timestep, denoised_actions):
            nonlocal cached_state_kv

            flow, cached_state_kv = self.forward(
                images,
                token_ids,
                joint_states,
                denoised_actions,
                times = timestep,
                reward_tokens = reward_tokens,
                cached_state_keys_values = cached_state_kv,
                return_actions_flow = True,
                return_state_keys_values = True
            )

            pbar.update(1)

            return flow

        # start with random gaussian noise - y0

        noise = torch.randn((batch_size, trajectory_length, self.dim_action_input), device = self.device)

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = self.odeint_fn(ode_fn, noise, times)

        sampled_actions = trajectory[-1]

        self.train(was_training)

        pbar.close()

        return sampled_actions

    def forward(
        self,
        images: Float['b nv d'] | Float['b c h w'] | Float['b c f h w'], # vision
        token_ids: Int['b nt'],                                          # language
        joint_state: Float['b djs'],                                     # joint state
        actions: Float['b na da'] | None = None,                         # action
        times: Float['b'] = None,
        reward_token: Float['b d'] | None = None,
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        **kwargs
    ):
        inferencing = exists(cached_state_keys_values)
        assert not (inferencing and not return_actions_flow), 'must be generating action trajectory if receiving cached state key values'

        if not exists(actions):
            return self.sample(images, token_ids, joint_state, **kwargs)

        batch, device = token_ids.shape[0], token_ids.device

        # noising the action for flow matching

        if not exists(times):
            times = torch.rand((batch,), device = device)

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # if not returning the actions predicted flow, assume training and noise the actions for loss

        if not return_actions_flow:
            noise = torch.randn_like(actions)

            flow = actions - noise
            padded_times = rearrange(times, 'b -> b 1 1')

            actions = noise * (1. - padded_times) + padded_times * actions

        # actions

        time_cond = self.to_time_cond(times)
        action_tokens = self.to_action_tokens(actions)

        if not inferencing:
            # language

            labels = token_ids[:, 1:]

            language_tokens = self.token_emb(token_ids)

            # vision

            if exists(self.vit):
                assert images.ndim in {4, 5}
                is_multiple_images = images.ndim == 5

                if is_multiple_images:
                    images = rearrange(images, 'b c f h w -> b f c h w')
                    images, images_frames_packed_shape = pack([images], '* c h w')

                with torch.no_grad():
                    self.vit.eval()                    
                    visual_tokens = self.vit(images)

                if is_multiple_images:
                    visual_tokens, = unpack(visual_tokens, images_frames_packed_shape, '* n d')
                    visual_tokens = rearrange(visual_tokens, 'b f n d -> b (f n) d')

            else:
                assert images.ndim == 3, 'images must be already encoded as (batch, seq, feature dimension)'
                visual_tokens = images

            visual_tokens = self.maybe_to_image_tokens(visual_tokens)

            # joint state

            joint_state_tokens = self.to_joint_state_tokens(joint_state)

            # maybe reward tokens

            if not exists(reward_tokens):
                reward_tokens = visual_tokens.new_empty((batch, 0, self.dim))

            # concat visual rep with language

            state_tokens, packed_shape = pack([visual_tokens, language_tokens, joint_state_tokens, reward_tokens], 'b * d')

        # prepare maybe flex attention

        flex_attn_fn = None

        if self.use_flex_attn and state_tokens.is_cuda:

            block_mask = None

            if not inferencing:
                prefix_length = state_tokens.shape[-2]
                seq_len = prefix_length + action_tokens.shape[-2]

                block_mask = create_block_mask(
                    create_pizero_attn_mask(prefix_length),
                    Q_LEN = seq_len,
                    KV_LEN = seq_len,
                    device = state_tokens.device
                )

            score_mod_fn = softclamp_score_mod(self.attn_softclamp_value)

            flex_attn_fn = partial(
                flex_attention,
                block_mask = block_mask,
                score_mod = score_mod
            )

        # state keys and values for caching during inference

        cached_state_key_values_iter = iter(default(cached_state_keys_values, []))

        state_cached_keys_values = []

        # value residual learning

        actions_value_residual = None

        # transformer

        if not inferencing:
            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                (state_attn_out, actions_attn_out), (state_keys, state_values, action_keys, action_values) = attn(state_tokens, action_tokens, flex_attn_fn = flex_attn_fn, actions_value_residual = actions_value_residual, return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                state_tokens = state_tokens + state_attn_out
                action_tokens = action_tokens + actions_attn_out

                state_tokens = state_ff(state_tokens) + state_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        else:

            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                actions_attn_out, (state_keys, state_values, action_keys, action_values) = attn.forward_actions_with_cached_state(action_tokens, cached_state_keys_values = next(cached_state_key_values_iter), return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                action_tokens = action_tokens + actions_attn_out

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        if not inferencing:
            # unpack and unembed to predictions

            visual_tokens, tokens, *_ = unpack(state_tokens, packed_shape, 'b * d')

            # gemma uses a final softclamp before norm

            tokens = self.final_norm_softclamp(tokens)

        action_tokens = self.final_norm_softclamp(action_tokens)

        # projection

        actions = self.final_actions_norm(action_tokens)

        # flow loss for actions tokens

        pred_actions_flow = self.actions_to_pred_flow(actions)

        if return_actions_flow:
            if not return_state_keys_values:
                return pred_actions_flow

            return pred_actions_flow, state_cached_keys_values

        flow_loss = F.mse_loss(flow, pred_actions_flow)

        # maybe direction loss

        dir_loss = self.zero

        if self.has_direction_loss:
            dir_loss = direction_loss(flow, pred_actions_flow)

        # language cross entropy loss

        tokens = self.final_norm(tokens)

        language_logits = self.state_to_logits(tokens)

        language_loss = F.cross_entropy(
            rearrange(language_logits[:, :-1], 'b n l -> b l n'),
            labels
        )

        # loss breakdown

        loss_breakdown = (language_loss, flow_loss, dir_loss)

        # total loss and return breakdown

        total_loss = (
            language_loss * self.lm_loss_weight +
            flow_loss * self.flow_loss_weight +
            dir_loss * self.direction_loss_weight
        )

        return total_loss, loss_breakdown

# fun

π0 = PiZero
