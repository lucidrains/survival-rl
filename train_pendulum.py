#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "einx>=0.4.3",
#     "einops>=0.8.2",
#     "fire",
#     "gymnasium",
#     "mean-conc-beta",
#     "memmap-replay-buffer",
#     "numpy",
#     "torch>=2.5",
#     "torch-einops-utils>=0.1.31",
#     "x-mlps-pytorch>=0.6.4",
# ]
# ///

from __future__ import annotations
from math import log, pi
import tempfile

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import gymnasium as gym
import einx
from einops import rearrange

from mean_conc_beta import Beta
from memmap_replay_buffer import ReplayBuffer
from survival_rl import HazardCritic, HazardCriticCompetitive, compute_first_dwell_time
from x_mlps_pytorch import MLP

# the goal is the full pendulum state, augmented with velocity and acceleration, so the base goal
# [1., 0., 0., 0.] is the upright equilibrium - dwelling at the goal then forces the policy to stabilize

GOAL = torch.tensor([1., 0., 0., 0.])
GOAL_DIM = GOAL.shape[-1]

def schedule_value(schedule, step):
    for until, value in schedule:
        if step <= until:
            return value
    return schedule[-1][1]

# actor

class Actor(nn.Module):
    def __init__(self, dim_state = 3, dim_goal = GOAL_DIM, dim_hidden = 128):
        super().__init__()
        self.dim_goal = dim_goal
        self.net = MLP(dim_state + dim_goal, dim_hidden, dim_hidden, 2, activation = nn.SiLU())
        self.beta = Beta(bounds = (-1., 1.), init_conc = 10., detach_entropy_mean = False)

    def dist(self, state, goal):
        x = torch.cat((state, goal[..., :self.dim_goal]), dim = -1)
        return self.beta(self.net(x).view(*x.shape[:-1], 1, 2))

    def forward(self, state, event):
        return self.dist(state, event).rsample()

    def entropy(self, state, event):
        return self.dist(state, event).entropy().sum(dim = -1)

# evaluation

@torch.no_grad()
def evaluate(actor, env, step, horizon, num_episodes = 10):
    device = next(actor.parameters()).device
    goal = GOAL.to(device)[None]
    scores = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        score = 0.

        for _ in range(horizon):
            state = torch.from_numpy(obs).float().to(device)[None]
            action = actor.dist(state, goal).mean[0]
            obs, reward, terminated, truncated, _ = step(action.cpu().numpy())
            score += reward

            if terminated or truncated:
                break

        scores.append(score)

    return float(np.mean(scores))

# main

def main(
    seed = 0,
    horizon = 200,
    num_envs = 16,
    batch_episodes = 8,
    grad_steps = 8,
    iterations = 1000,
    eval_every = 25,
    warmup = 3,
    target_return = -250.,
    dwell_schedule = ((40, 1), (90, 2), (300, 3), (500, 6), (1000, 10)),
    epsilon_schedule = ((250, 0.3), (1000, 0.1)),
    entropy_schedule = ((400, 0.5), (1000, 0.1)),
    eps = 0.25,
    vel_scale = 8.,
    acc_scale = 21.,
    dt = 0.05,
    gamma = 0.99,
    lr_actor = 3e-4,
    lr_critic = 1e-3,
    p_base = 0.35,
    p_future = 0.35,
    p_random = 0.15,
    collect_base_prob = 0.5,
    competitive = False
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_dwell = dwell_schedule[-1][1]

    actor = Actor().to(device)
    critic_klass = HazardCriticCompetitive if competitive else HazardCritic
    critic = critic_klass(
        dim = 128,
        depth = 2,
        dim_state = 3,
        num_actions = 1,
        pred_time_bins = horizon,
        dim_event = GOAL_DIM + 1,
        discount_factor = gamma
    ).to(device)

    actor_optim = AdamW(actor.parameters(), lr = lr_actor)
    critic_optim = AdamW(critic.parameters(), lr = lr_critic)

    log_alpha = nn.Parameter(torch.tensor(log(0.02), device = device))
    alpha_optim = AdamW([log_alpha], lr = lr_actor)

    temp_dir = tempfile.TemporaryDirectory()
    buffer = ReplayBuffer(
        temp_dir.name,
        max_episodes = 256,
        max_timesteps = horizon,
        fields = dict(obs = ('float', 3), action = ('float', 1)),
        circular = True,
        overwrite = True
    )

    envs = [gym.make('Pendulum-v1') for _ in range(num_envs)]
    eval_env = gym.make('Pendulum-v1')

    step_fns = [actor.beta.rescale_env_step(e.step, target_range = (-2., 2.), clip = True) for e in envs]
    eval_step = actor.beta.rescale_env_step(eval_env.step, target_range = (-2., 2.), clip = True)

    base_goal = GOAL.to(device)

    # pendulum observation to goal space - angle, scaled velocity, and finite difference acceleration

    def project(obs):
        velocity = obs[..., 2:]
        acceleration = torch.zeros_like(velocity)
        acceleration[:, :-1] = (velocity[:, 1:] - velocity[:, :-1]) / dt
        acceleration[:, -1] = acceleration[:, -2]

        return torch.cat((obs[..., :2], velocity / vel_scale, acceleration / acc_scale), dim = -1)

    # survival relabeling - each transition is assigned a goal, and the event is the first dwell at that goal

    def relabel(obs, action, dwell):
        batch, timesteps, _ = obs.shape
        valid = timesteps - dwell
        times = torch.arange(timesteps, device = device)
        states = project(obs)

        time_delta = times[None, :] - times[:, None]
        weights = (gamma ** time_delta.clamp(min = 0.)).masked_fill(time_delta < 1, 0.)
        weights = weights + (weights.sum(-1, keepdim = True) <= 0.) / timesteps
        future_idx = torch.multinomial(weights, 1)[:, 0].clamp(max = timesteps - 1)

        random_idx = torch.randint(timesteps, (batch, timesteps), device = device)
        random_goal = einx.get_at('b [t] d, b q -> b q d', states, random_idx)
        future_goal = einx.get_at('b [t] d, b q -> b q d', states, future_idx.expand(batch, -1))
        base = base_goal.expand(batch, timesteps, GOAL_DIM)

        coin = torch.rand(batch, timesteps, device = device)
        goal = torch.where((coin < p_base)[..., None], base,
               torch.where((coin < p_base + p_future)[..., None], future_goal,
               torch.where((coin < p_base + p_future + p_random)[..., None], random_goal, states)))

        reach_index, cutoff = compute_first_dwell_time(goal[:, :valid], states, eps = eps, dwell_steps = dwell)

        dwell_feature = torch.full((batch, valid, 1), dwell / max_dwell, device = device)
        event = torch.cat((goal[:, :valid], dwell_feature), dim = -1)

        return (
            rearrange(obs[:, :valid], 'b t d -> (b t) d'),
            rearrange(action[:, :valid], 'b t d -> (b t) d'),
            rearrange(event, 'b t d -> (b t) d'),
            reach_index.reshape(-1),
            cutoff.reshape(-1)
        )

    # collect - half of the environments chase the upright equilibrium, the rest chase a random angle

    def collect(epsilon, warmup = False):
        obs = np.stack([e.reset()[0] for e in envs])

        is_base = np.random.rand(num_envs) < collect_base_prob
        angles = np.random.uniform(-pi, pi, num_envs)

        goals = np.zeros((num_envs, GOAL_DIM), dtype = np.float32)
        goals[:, 0] = np.cos(angles)
        goals[:, 1] = np.sin(angles)
        goals = np.where(is_base[:, None], base_goal.cpu().numpy(), goals)
        goals = torch.tensor(goals, dtype = torch.float32, device = device)

        with buffer.batched_episode(batch_size = num_envs):
            for _ in range(horizon):
                if warmup:
                    actions = torch.empty(num_envs, 1, device = device).uniform_(-1., 1.)
                else:
                    actions = actor.dist(torch.from_numpy(obs).float().to(device), goals).sample()
                    is_random = torch.rand(num_envs, 1, device = device) < epsilon
                    actions = torch.where(is_random, torch.empty_like(actions).uniform_(-1., 1.), actions)

                buffer.store_batch(obs = torch.from_numpy(obs), action = actions.cpu())

                for i, step_fn in enumerate(step_fns):
                    obs[i] = step_fn(actions[i].cpu().numpy())[0]

    # update - maximum likelihood hazard critic, then a sac actor maximizing the survival value

    def update(obs, action, dwell, target_entropy):
        obs, action = obs.to(device), action.to(device)
        states, actions, events, reach_index, cutoff = relabel(obs, action, dwell)

        critic_optim.zero_grad()
        critic_loss = critic(states, actions, events, reach_event_index = reach_index, horizon_cutoff = cutoff)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        critic_optim.step()

        dist = actor.dist(states, events)
        actions_new = dist.rsample()
        values = critic(states, actions_new, events, return_values = True)
        entropy = dist.entropy().sum(dim = -1)

        alpha = log_alpha.exp()

        actor_optim.zero_grad()
        actor_loss = (-alpha.detach() * entropy - values).mean()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        actor_optim.step()

        alpha_optim.zero_grad()
        alpha_loss = (alpha * (entropy.detach() - target_entropy)).mean()
        alpha_loss.backward()
        alpha_optim.step()

        return critic_loss.item(), values.mean().item()

    score = evaluate(actor, eval_env, eval_step, horizon)

    for iteration in range(1, iterations + 1):
        dwell = schedule_value(dwell_schedule, iteration)
        epsilon = schedule_value(epsilon_schedule, iteration)
        target_entropy = schedule_value(entropy_schedule, iteration)

        collect(epsilon, warmup = iteration <= warmup)

        loader = buffer.dataloader(batch_size = batch_episodes, to_named_tuple = ('obs', 'action'), shuffle = True)
        losses = [update(batch.obs, batch.action, dwell, target_entropy) for _, batch in zip(range(grad_steps), loader)]

        if iteration % eval_every == 0 or iteration == iterations:
            score = evaluate(actor, eval_env, eval_step, horizon)
            print(f'iteration {iteration:4d} | dwell {dwell:2d} | critic loss {np.mean([l[0] for l in losses]):.4f} | value {np.mean([l[1] for l in losses]):.2f} | eval return {score:.1f}', flush = True)

            if score >= target_return:
                break

    for env in envs:
        env.close()
    eval_env.close()
    temp_dir.cleanup()

    assert score >= target_return, f'expected the inverted pendulum to balance, final eval return: {score:.1f}'
    print(f'inverted pendulum balanced with survival RL (eval return: {score:.1f})')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
