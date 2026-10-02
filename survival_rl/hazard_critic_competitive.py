from __future__ import annotations
from math import log

import torch
from torch.nn.functional import log_softmax
from torch.nn import Module, Linear, SiLU

from torch_einops_utils import batched_index_select

from x_mlps_pytorch import (
    MLP,
    ResidualNormedMLP,
    AttnResidualNormedMLP
)

from survival_rl.survival_rl import CriticOutput

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class HazardCriticCompetitive(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_state,
        num_actions,
        pred_time_bins,
        dim_action = None,
        dim_event = None,
        encode_event = True,
        actor_event_kwarg = 'event',
        attn_residual = True,
        discount_factor = 0.99
    ):
        super().__init__()

        self.actor_event_kwarg = actor_event_kwarg

        mlp_klass = AttnResidualNormedMLP if attn_residual else ResidualNormedMLP

        dim_event = default(dim_event, dim_state)
        dim_action = default(dim_action, dim // 2)

        self.action_encoder = MLP(num_actions, dim_action, dim_action)

        # encode event / goal before film, rather than a linear projection

        event_encoder = MLP(dim_event, dim, dim, activation = SiLU()) if encode_event else None

        mlp_kwargs = dict(
            dim_in = dim_state + dim_action,
            dim = dim,
            depth = depth,
            film = True,
            cond_dim = dim if encode_event else dim_event,
            cond_encoder = event_encoder
        )

        if not attn_residual:
            mlp_kwargs['residual_every'] = next(e for e in (4, 3, 2, 1) if depth % e == 0)

        self.mlp = mlp_klass(**mlp_kwargs)

        self.num_time_bins = pred_time_bins

        # one extra bucket holds the right censored / beyond horizon mass, and participates in every tail

        self.pred_time_bins = Linear(dim, pred_time_bins + 1)

        # calculate log discount factors upfront

        self.register_buffer('log_discounts', torch.arange(pred_time_bins) * log(discount_factor))

    def extract_policy(
        self,
        actor: Module,
        state,
        event,
        negate = False
    ):
        actions = actor(state, **{self.actor_event_kwarg: event})
        values = self.forward(state, actions, event, return_values = True)

        # negate for avoiding an adverse event

        return -values if negate else values

    def forward(
        self,
        state,
        actions,
        event,
        reach_event_index = None, # -1 or >= horizon_cutoff treated as right-censored
        horizon_cutoff = None,
        return_values = False
    ):
        horizon_cutoff = default(horizon_cutoff, self.num_time_bins)
        batch, device = state.shape[0], state.device

        # encode actions, embed with deep mlp and predict bins

        encoded_action = self.action_encoder(actions)

        embed = self.mlp((state, encoded_action), cond = event)

        time_bin_logits = self.pred_time_bins(embed)

        # competing bins - softmax makes the bins compete for a single unit of mass

        log_probs = log_softmax(time_bin_logits, dim = -1)

        # log_tail[..., i] = log P(T >= i), the final bucket folded into every earlier tail

        log_tail = log_probs.flip(dims = (-1,)).logcumsumexp(dim = -1).flip(dims = (-1,))
        log_survival = log_tail[..., 1:]

        # returning values, as discounted sum of survival probabilities

        if return_values:
            return -(log_survival + self.log_discounts).exp().sum(dim = -1)

        # if no reach event index given, return the raw logits alongside the log survival function

        if not exists(reach_event_index):
            return CriticOutput(time_bin_logits, log_survival)

        # uncensored - plain cross entropy on the observed event bin

        hit_bin = reach_event_index.clamp(min = 0, max = self.num_time_bins)
        hit_loss = -batched_index_select(log_probs, hit_bin)

        # right censored - cross entropy against all bins at or past the cutoff, including the final bucket
        # when censoring always happens at the horizon, this reduces to the final bucket cross entropy

        tail_bin = torch.as_tensor(horizon_cutoff, device = device).clamp(min = 0, max = self.num_time_bins)
        tail_bin = tail_bin.expand(batch)
        censored_loss = -batched_index_select(log_tail, tail_bin)

        reached_event = (reach_event_index >= 0) & (reach_event_index < horizon_cutoff)
        return torch.where(reached_event, hit_loss, censored_loss).mean()
