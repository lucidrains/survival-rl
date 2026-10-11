from __future__ import annotations
from math import log
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import (
    logsigmoid,
    binary_cross_entropy_with_logits
)
from torch.nn import Module, Linear, SiLU

import einx
from einops import rearrange

from torch_einops_utils import (
    lens_to_mask,
    masked_sum,
    pad_left_at_dim,
    pad_right_ndim_to,
    exclusive_cumsum,
    reverse_cumsum,
    safe_cat
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

# bellman bootstrap on censored transitions

# per bin hazards (b h) -> event pmf (b h + 1), tail bucket holds the mass past the horizon

def hazard_logits_to_event_probs(logits):
    log_hazard, log_survival = logsigmoid(logits), logsigmoid(-logits)
    event_probs = (log_hazard + exclusive_cumsum(log_survival, dim = -1)).exp()
    tail = log_survival.sum(dim = -1, keepdim = True).exp()
    return safe_cat((event_probs, tail), dim = -1)

# nll of the per bin hazards against a soft event pmf, one bin per logit

def soft_target_nll(logits, target_probs):
    event_target = target_probs[..., :-1]
    survival_target = reverse_cumsum(target_probs, dim = -1)[..., 1:]
    return -(event_target * logsigmoid(logits) + survival_target * logsigmoid(-logits)).sum(dim = -1)

# shift event times right by steps, folding the overflow past the horizon into the tail bucket

def shift_event_probs(probs, steps):
    num_bins = probs.shape[-1] - 1

    steps = torch.as_tensor(steps, device = probs.device)
    indices = einx.add('h, ... -> ... h', torch.arange(num_bins, device = probs.device), -steps)

    event_probs = probs[..., :num_bins]
    shifted = event_probs.gather(-1, indices.clamp(min = 0)).masked_fill(indices < 0, 0.)

    overflow = (event_probs.sum(dim = -1) - shifted.sum(dim = -1)).clamp(min = 0.)
    return safe_cat((shifted, probs[..., -1:] + overflow[..., None]), dim = -1)

# first time the trajectory dwells at the goal

def compute_first_dwell_time(
    goal,               # (..., num_queries, dim)
    states = None,      # (..., num_steps + 1, dim) - states the actions were taken from
    next_states = None, # (..., num_steps, dim) - post-action states (s_1, s_2, ...)
    eps = 0.25,
    dwell_steps = 1,    # int or tensor of dwell steps
    lens = None         # int or tensor of trajectory lengths
):
    """
    computes the first index the trajectory dwells at the goal
    one of states or next_states must be given

    returns the reach index (-1 if never reached) and the cutoff, both (..., num_queries)
    """
    assert exists(states) or exists(next_states), 'either states or next_states must be given'

    if not exists(next_states):
        next_states = states[..., 1:, :]

    device = goal.device
    num_queries, num_steps = goal.shape[-2], next_states.shape[-2]

    # whether each post-action state lies within eps of each goal

    inside = einx.subtract('... q d, ... t d -> ... q t d', goal, next_states).norm(dim = -1) <= eps

    # dwell steps and lengths as tensor padded to match dimensions

    lens = default(lens, num_steps)
    lens = torch.as_tensor(lens, device = device)
    lens = pad_right_ndim_to(lens, inside.ndim - 1)
    padded_lens = pad_right_ndim_to(lens, inside.ndim)

    k = torch.as_tensor(dwell_steps, device = device)
    k = pad_right_ndim_to(k, inside.ndim - 1)
    padded_k = pad_right_ndim_to(k, inside.ndim)

    # mask out post-action states past sequence length

    u = torch.arange(num_steps, device = device)
    times = torch.arange(num_queries, device = device)

    inside = inside & (u < padded_lens)

    # cumulative sum for sliding window count of in-goal states

    counts = pad_left_at_dim(inside.int().cumsum(dim = -1), 1)

    valid_window = (u + padded_k) <= padded_lens
    end_idx = (u + padded_k).clamp(max = num_steps).expand_as(inside)

    sum_window = counts.gather(-1, end_idx) - counts[..., :-1]
    dwell = valid_window & (sum_window == padded_k)

    # first dwell window starting at or after each transition

    candidate = dwell & (u >= rearrange(times, 'q -> q 1'))

    reached = candidate.any(dim = -1)
    first_start = candidate.long().argmax(dim = -1)

    reach_index = torch.where(reached, first_start - times, -1)

    cutoff = lens - k + 1 - times
    cutoff = cutoff.clamp(min = 0).expand_as(reach_index)

    return reach_index, cutoff

# outputs

class CriticOutput(NamedTuple):
    logits: Tensor
    log_survival: Tensor

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
        dim_event = None,
        dim_event_encoded = None,
        encode_event = True,
        event_encoder = None,
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

        if exists(event_encoder):
            encode_event = True
        elif encode_event:
            event_encoder = MLP(dim_event, dim, dim, activation = SiLU())

        mlp_kwargs = dict(
            dim_in = dim_state + dim_action,
            dim = dim,
            depth = depth,
            film = True,
            cond_dim = default(dim_event_encoded, dim if encode_event else dim_event),
            cond_encoder = event_encoder
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
        return_values = False,
        bootstrap_state = None,   # state at t + steps, for censored transitions
        bootstrap_action = None,  # action taken at t + steps
        bootstrap_steps = None,   # certified failed dwell windows, 0 falls back to the censor tail
        bootstrap_critic = None   # target critic for the shifted event pmf, defaults to self
    ):
        horizon_cutoff = default(horizon_cutoff, self.num_time_bins)

        # encode actions, embed with deep mlp and predict bins

        encoded_action = self.action_encoder(actions)

        embed = self.mlp((state, encoded_action), cond = event)

        time_bin_logits = self.pred_time_bins(embed)

        # log survival

        log_one_minus_hazard = logsigmoid(-time_bin_logits)

        log_survival = log_one_minus_hazard.cumsum(dim = -1)

        # returning values, as discounted sum of survival probabilities

        if return_values:
            return -(log_survival + self.log_discounts).exp().sum(dim = -1)

        # if no reach event index given, return the raw logits alongside the log survival function

        if not exists(reach_event_index):
            return CriticOutput(time_bin_logits, log_survival)

        # bce loss on logits

        reached_event = rearrange(reach_event_index < horizon_cutoff, 'b -> b 1')
        reached_event_bin = einx.equal('b, t -> b t', reach_event_index, self.times) & reached_event

        losses = binary_cross_entropy_with_logits(time_bin_logits, reached_event_bin.to(time_bin_logits.dtype), reduction = 'none')

        # nll loss - time after cutoff is masked out

        lens = torch.where(reach_event_index >= 0, reach_event_index + 1, horizon_cutoff).clamp(max = horizon_cutoff)
        mask = lens_to_mask(lens, self.num_time_bins)

        loss = masked_sum(losses, mask, dim = -1)

        # optional bellman bootstrap on censored transitions

        if exists(bootstrap_steps):
            assert exists(bootstrap_state) and exists(bootstrap_action)

            with torch.no_grad():
                target_critic = default(bootstrap_critic, self)
                target_logits = target_critic(bootstrap_state, bootstrap_action, event).logits
                target_event_probs = shift_event_probs(hazard_logits_to_event_probs(target_logits), bootstrap_steps)

            observed = (reach_event_index >= 0) & (reach_event_index < horizon_cutoff)
            use_bootstrap = ~observed & (bootstrap_steps > 0)

            loss = torch.where(use_bootstrap, soft_target_nll(time_bin_logits, target_event_probs), loss)

        return loss.mean()
