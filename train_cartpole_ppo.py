#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "env-ssl-wrapper>=0.5.15",
#     "fire",
#     "gymnasium",
#     "numpy",
#     "survival-rl",
#     "torch>=2.5",
#     "x-mlps-pytorch>=0.6.4",
#     "x-ppo>=0.0.14",
# ]
# ///

# PPO with a survival (hazard) critic on CartPole.
#
# event = episode end. the critic models p(time to episode end | state, policy),
# and its discounted survival sum is the value of the finite-horizon return.
# the critic is fit by hazard nll, the actor by clipped ppo.

from __future__ import annotations

import fire
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from env_ssl_wrapper import StandardizeEnvWrapper, evaluate_actor
from survival_rl import HazardCritic
from torch_einops_utils import shift_right
from x_mlps_pytorch import MLP
from x_ppo import calc_gae, ppo_actor_loss

# actor

class Actor(nn.Module):
    def __init__(self, dim_hidden = 64, num_actions = 2):
        super().__init__()
        self.register_buffer('scale', torch.tensor([2.4, 3.0, 0.21, 3.0]))
        self.net = MLP(4, dim_hidden, dim_hidden, num_actions, activation = nn.SiLU())

    def forward(self, state):
        state = (state / self.scale).clamp(-10., 10.)
        return self.net(state).softmax(dim = -1)  # (b 4) -> (b 2)

# env

def make_env(num_envs, device):
    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)])
    return StandardizeEnvWrapper(env, device = device, done_tracker = False, pad_episodes = False)

# value readout

def hazard_value(critic, state, prob, event):
    return -critic(state, prob, event, return_values = True)

@torch.no_grad()
def collect(env, actor, critic, event, horizon):
    obs, _ = env.reset()
    buf = dict(state = [], prob = [], action = [], logp = [], reward = [], done = [])

    for _ in range(horizon):
        prob = actor(obs)
        dist = Categorical(prob)
        action = dist.sample()

        for key, val in (('state', obs), ('prob', prob), ('action', action), ('logp', dist.log_prob(action))):
            buf[key].append(val)

        obs, reward, terminated, truncated, _ = env.step(action)
        buf['reward'].append(reward)
        buf['done'].append(terminated | truncated)

    out = {k: torch.stack(v, dim = 1) for k, v in buf.items()}
    flat_state, flat_prob = out['state'].reshape(-1, 4), out['prob'].reshape(-1, 2)
    flat_event = event[:, None].expand(-1, horizon, -1).reshape(-1, 4)
    out['value'] = hazard_value(critic, flat_state, flat_prob, flat_event).reshape(-1, horizon)
    next_value = hazard_value(critic, obs, actor(obs), event)

    return out, next_value

# critic targets - remaining steps to episode end, -1 for a partial rollout tail

def episode_targets(done):
    horizon = done.shape[1]
    pos = torch.arange(horizon, device = done.device)
    start = torch.cummax(torch.where(shift_right(done, 1, pad_value = True), pos, -1), dim = -1).values
    end = torch.where(done, pos, horizon)
    end = torch.flip(torch.cummin(torch.flip(end, dims = (-1,)), dim = -1).values, dims = (-1,))
    target = torch.where(end < horizon, end - pos + 1, -1)
    lengths = target[(pos == start) & (target > 0)]
    return target, lengths

# main

def main(
    seed = 0,
    device = 'cpu',
    num_envs = 16,
    max_steps = 500,
    num_updates = 20,
    eval_every = 2,
    eval_episodes = 10,
    target_return = 475.,
    gamma = 0.99,
    lam = 0.95,
    clip = 0.2,
    actor_lr = 3e-4,
    critic_lr = 1e-3,
    actor_epochs = 4,
    critic_epochs = 1,
    minibatch = 512,
    entropy_coef = 0.01,
    dim_hidden = 64,
    critic_dim = 64,
    critic_depth = 2
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)
    env = make_env(num_envs, device)
    eval_env = StandardizeEnvWrapper(gym.make('CartPole-v1'), device = device, done_tracker = False, pad_episodes = False)

    actor = Actor(dim_hidden).to(device)
    critic = HazardCritic(
        dim = critic_dim,
        depth = critic_depth,
        dim_state = 4,
        num_actions = 2,
        dim_event = 4,
        pred_time_bins = max_steps,
        discount_factor = gamma
    ).to(device)

    actor_optim = Adam(actor.parameters(), lr = actor_lr)
    critic_optim = Adam(critic.parameters(), lr = critic_lr)
    event = torch.zeros(num_envs, 4, device = device)

    train_returns, critic_losses = [], []
    score = 0.

    print(f'device {device} | {num_envs} envs | {max_steps} hazard bins', flush = True)

    for update in range(1, num_updates + 1):
        batch, next_value = collect(env, actor, critic, event, max_steps)

        gae = calc_gae(
            batch['reward'], batch['value'],
            masks = (~batch['done']).float(),
            gamma = gamma,
            lam = lam,
            next_value = next_value,
            return_advantages = True
        )

        adv = gae.advantages.reshape(-1)
        target, lengths = episode_targets(batch['done'])
        train_returns.append(lengths.float().mean().item())

        state = batch['state'].reshape(-1, 4)
        prob = batch['prob'].reshape(-1, 2)
        action = batch['action'].reshape(-1)
        old_logp = batch['logp'].reshape(-1)
        target = target.reshape(-1)
        cutoff = torch.where(target >= 0, max_steps, 0)
        event_flat = event[:, None].expand(-1, max_steps, -1).reshape(-1, 4)

        # critic - hazard nll

        for _ in range(critic_epochs):
            perm = torch.randperm(state.shape[0], device = device)

            for start in range(0, state.shape[0], minibatch):
                idx = perm[start:start + minibatch]
                loss = critic(state[idx], prob[idx], event_flat[idx], reach_event_index = target[idx], horizon_cutoff = cutoff[idx])

                critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                critic_optim.step()
                critic_losses.append(loss.item())

        # actor - clipped ppo

        for _ in range(actor_epochs):
            perm = torch.randperm(state.shape[0], device = device)

            for start in range(0, state.shape[0], minibatch):
                idx = perm[start:start + minibatch]
                dist = Categorical(actor(state[idx]))
                entropy = dist.entropy().mean()
                loss = ppo_actor_loss(dist.log_prob(action[idx]), old_logp[idx], adv[idx], eps_clip = clip, normalize_advantages = True).mean()
                loss = loss - entropy_coef * entropy

                actor_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                actor_optim.step()

        if update % eval_every == 0 or update == 1:
            score = evaluate_actor(actor, eval_env, episodes = eval_episodes, max_steps = max_steps, seed = seed).mean

            print(
                f'update {update:3d} | steps {update * num_envs * max_steps:7d} | '
                f'train return {np.mean(train_returns[-5:]):5.1f} | eval {score:5.1f} | '
                f'critic loss {np.mean(critic_losses[-64:]):.3f}',
                flush = True
            )

            if score >= target_return:
                print(f'cartpole solved at update {update} with survival-value PPO (eval {score:.1f})', flush = True)
                break

    env.close()
    eval_env.close()

if __name__ == '__main__':
    fire.Fire(main)
