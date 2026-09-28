from __future__ import annotations
from math import log

import torch
from torch.nn.functional import (
    logsigmoid,
    binary_cross_entropy_with_logits
)
from torch.nn import Module, Linear

import einx

from torch_einops_utils import (
    lens_to_mask,
    masked_sum
)

from x_mlps_pytorch import (
    MLP,
    ResidualNormedMLP,
    AttnResidualNormedMLP
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class HazardCritic(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_state,
        num_actions,
        pred_time_bins,
        dim_action = None,
        dim_goal = None,
        attn_residual = True,
        discount_factor = 0.99
    ):
        super().__init__()

        mlp_klass = AttnResidualNormedMLP if attn_residual else ResidualNormedMLP

        dim_goal = default(dim_goal, dim_state)
        dim_action = default(dim_action, dim // 2)

        self.action_encoder = MLP(num_actions, dim_action, dim_action)

        mlp_kwargs = dict(
            dim_in = dim_state + dim_goal + dim_action,
            dim = dim,
            depth = depth
        )

        if not attn_residual:
            mlp_kwargs['residual_every'] = next(e for e in (4, 3, 2, 1) if depth % e == 0)

        self.mlp = mlp_klass(**mlp_kwargs)

        self.pred_time_bins = Linear(dim, pred_time_bins)

        self.num_time_bins = pred_time_bins

        self.register_buffer('times', torch.arange(pred_time_bins))

        # calculate log discount factors upfront

        self.register_buffer('log_discounts', torch.arange(pred_time_bins) * log(discount_factor))

    def extract_policy(
        self,
        actor: Module,
        state,
        goal,
    ):
        actions = actor(state, goal)
        values = self.forward(state, actions, goal, return_values = True)
        return -values

    def forward(
        self,
        state,
        actions,
        goal,
        reach_goal_index = None, # -1 or >= horizon_cutoff treated as right-censored
        horizon_cutoff = None,
        return_values = False
    ):
        horizon_cutoff = default(horizon_cutoff, self.num_time_bins)

        # encode actions, embed with deep mlp and predict bins

        encoded_action = self.action_encoder(actions)

        embed = self.mlp((state, encoded_action, goal))

        time_bin_logits = self.pred_time_bins(embed)

        # log survival

        log_one_minus_hazard = logsigmoid(-time_bin_logits)

        log_survival = log_one_minus_hazard.cumsum(dim = -1)

        # returning values, as discounted sum of survival probabilities

        if return_values:
            return -(log_survival + self.log_discounts).exp().sum(dim = -1)

        # if no reach goal index given, return the log survival function

        if not exists(reach_goal_index):
            return log_survival

        # bce loss on logits

        reached_goal = reach_goal_index < horizon_cutoff
        reached_goal_bin = einx.equal('b, t -> b t', reach_goal_index, self.times) & reached_goal[..., None]

        losses = binary_cross_entropy_with_logits(time_bin_logits, reached_goal_bin.to(time_bin_logits.dtype), reduction = 'none')

        # nll loss - time after cutoff is masked out

        lens = torch.where(reach_goal_index >= 0, reach_goal_index + 1, horizon_cutoff).clamp(max = horizon_cutoff)
        mask = lens_to_mask(lens, self.num_time_bins)

        return masked_sum(losses, mask, dim = -1).mean()
