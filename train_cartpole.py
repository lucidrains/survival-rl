#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "einx>=0.4.3",
#     "einops>=0.8.2",
#     "env-ssl-wrapper>=0.5.0",
#     "fire",
#     "gymnasium",
#     "memmap-replay-buffer",
#     "numpy",
#     "survival-rl",
#     "torch>=2.5",
#     "torch-einops-utils>=0.1.31",
#     "x-mlps-pytorch>=0.6.4",
# ]
# ///

from __future__ import annotations
from math import log
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
import gymnasium as gym
import einx
from einops import repeat

from env_ssl_wrapper import evaluate_actor
from memmap_replay_buffer import ReplayBuffer
from survival_rl import HazardCritic, compute_first_dwell_time
from torch_einops_utils import lens_to_mask
from x_mlps_pytorch import MLP

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def schedule_value(schedule, step):
    for until, value in schedule:
        if step <= until:
            return value
    return schedule[-1][1]

# the goal is the balanced pole at the center of the track, with states scaled
# so the eps-ball around the goal is the region the pole must dwell in

GOAL = torch.tensor([0., 0., 0., 0.])
GOAL_DIM = GOAL.shape[-1]
STATE_SCALE = torch.tensor([2.4, 3.0, 0.21, 3.0])

DWELL_SCHEDULE = ((40, 1), (90, 2), (200, 4), (350, 8), (500, 12))
EPSILON_SCHEDULE = ((150, 0.25), (500, 0.05))
ENTROPY_SCHEDULE = ((250, 0.4), (500, 0.1))

# actor - the critic consumes a continuous representation of the categorical action distribution,
# so policy extraction can differentiate through it

class Actor(nn.Module):
    """
    action_repr
      'softmax'    - action probabilities (https://arxiv.org/abs/2509.10656)
      'raw_logits' - raw logits
      'gumbel'     - gumbel-softmax sample, the naive reparameterization
    """
    def __init__(
        self,
        dim_state = 4,
        dim_goal = GOAL_DIM,
        dim_hidden = 128,
        num_actions = 2,
        action_repr = 'softmax',
        gumbel_tau = 1.,
        gumbel_hard = False
    ):
        super().__init__()

        assert action_repr in ('softmax', 'raw_logits', 'gumbel')

        self.dim_goal = dim_goal
        self.num_actions = num_actions
        self.action_repr = action_repr
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard

        self.net = MLP(dim_state + dim_goal, dim_hidden, dim_hidden, num_actions, activation = nn.SiLU())

    def logits(self, state, goal):
        x = torch.cat((state, goal[..., :self.dim_goal]), dim = -1)
        return self.net(x)   # (b, num_actions)

    def dist(self, state, goal):
        return Categorical(logits = self.logits(state, goal))

    def forward(self, state, goal):
        logits = self.logits(state, goal)

        if self.action_repr == 'raw_logits':
            return logits

        if self.action_repr == 'gumbel':
            return F.gumbel_softmax(logits, tau = self.gumbel_tau, hard = self.gumbel_hard)   # jang et al. 2017

        return logits.softmax(dim = -1)

# main

def main(
    seed = 0,
    num_envs = 16,
    horizon = 500,
    iterations = 500,
    eval_every = 10,
    num_eval_episodes = 10,
    warmup = 3,
    target_return = 475.,
    action_repr = 'softmax',
    gumbel_tau = 1.,
    gumbel_hard = False,
    eps = 0.25,
    gamma = 0.99,
    batch_episodes = 8,
    grad_steps = 8,
    lr_actor = 3e-4,
    lr_critic = 1e-3,
    p_base = 0.35,
    p_future = 0.35,
    p_random = 0.15,
    collect_base_prob = 0.5,
    checkpoint_file = 'cartpole_best.pt',
    device = ''
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    state_scale = STATE_SCALE.to(device)
    base_goal = GOAL.to(device)

    # env

    envs = [gym.make('CartPole-v1') for _ in range(num_envs)]
    eval_env = gym.make('CartPole-v1')

    # replay buffer

    temp_dir = tempfile.TemporaryDirectory()
    buffer = ReplayBuffer(
        temp_dir.name,
        max_episodes = 256,
        max_timesteps = horizon,
        fields = dict(obs = ('float', 4), action = ('float', 2)),
        circular = True,
        overwrite = True
    )

    # models

    actor = Actor(
        action_repr = action_repr,
        gumbel_tau = gumbel_tau,
        gumbel_hard = gumbel_hard
    ).to(device)

    critic = HazardCritic(
        dim = 128,
        depth = 2,
        dim_state = 4,
        num_actions = 2,
        pred_time_bins = horizon,
        dim_event = GOAL_DIM + 1,
        discount_factor = gamma
    ).to(device)

    # optimizers

    actor_optim = AdamW(actor.parameters(), lr = lr_actor)
    critic_optim = AdamW(critic.parameters(), lr = lr_critic)

    log_alpha = nn.Parameter(torch.tensor(log(0.02), device = device))
    alpha_optim = AdamW([log_alpha], lr = lr_actor)

    # collect - environments chase the balanced state half the time and a random state otherwise

    stats = dict()

    def collect(epsilon, warmup = False):
        obs = [env.reset()[0] for env in envs]

        random_goals = np.random.uniform(-1., 1., (num_envs, GOAL_DIM)).astype(np.float32) * state_scale.cpu().numpy()
        is_base = np.random.rand(num_envs) < collect_base_prob
        goals = torch.tensor(np.where(is_base[:, None], 0., random_goals), dtype = torch.float32, device = device)

        trajectories = [dict(obs = [], action = [], returns = 0.) for _ in range(num_envs)]
        active = list(range(num_envs))

        while active:
            idx = torch.tensor(active, device = device)
            states = torch.as_tensor(np.stack([obs[i] for i in active]), dtype = torch.float32, device = device)

            with torch.no_grad():
                if warmup:
                    actions = torch.randint(0, 2, (len(active),), device = device)
                    action_repr = F.one_hot(actions, 2).float()
                else:
                    action_repr = actor(states, goals[idx])

                    # gumbel-max turns the reparameterized sample into an exact categorical sample

                    actions = action_repr.argmax(dim = -1) if actor.action_repr == 'gumbel' \
                        else actor.dist(states, goals[idx]).sample()

                    is_random = torch.rand(len(active), device = device) < epsilon
                    if is_random.any():
                        random_actions = torch.randint(0, 2, actions.shape, device = device)
                        random_repr = F.one_hot(random_actions, 2).float()
                        actions = torch.where(is_random, random_actions, actions)
                        action_repr = torch.where(is_random[:, None], random_repr, action_repr)

            actions_np = actions.cpu().numpy()
            action_repr_np = action_repr.cpu().numpy()

            for j, i in enumerate(list(active)):
                trajectory = trajectories[i]
                trajectory['obs'].append(np.asarray(obs[i], dtype = np.float32))
                trajectory['action'].append(action_repr_np[j])

                next_obs, reward, terminated, truncated, _ = envs[i].step(int(actions_np[j]))
                obs[i] = next_obs
                trajectory['returns'] += reward

                if terminated or truncated or len(trajectory['obs']) >= horizon:
                    if len(trajectory['obs']) >= 2:
                        buffer.store_episode(
                            obs = np.stack(trajectory['obs']),
                            action = np.stack(trajectory['action'])
                        )

                    active.remove(i)

        stats['collect_return'] = float(np.mean([t['returns'] for t in trajectories]))

    # survival relabeling - each transition is assigned a goal, and the event is the first dwell at that goal

    def relabel(obs, action, lens, dwell):
        batch, timesteps, _ = obs.shape
        lens = lens.long()
        times = torch.arange(timesteps, device = device)
        states = obs / state_scale

        time_delta = times[None, :] - times[:, None]
        weights = (gamma ** time_delta.clamp(min = 0.)).masked_fill(time_delta < 1, 0.)
        weights = weights + (weights.sum(-1, keepdim = True) <= 0.) / timesteps
        future_idx = torch.multinomial(weights, 1)[:, 0]
        future_idx = repeat(future_idx, 't -> b t', b = batch).clamp(max = (lens - 1).clamp(min = 0)[:, None])

        future_goal = einx.get_at('b [t] d, b q -> b q d', states, future_idx)

        random_idx = torch.randint(timesteps, (batch, timesteps), device = device).clamp(max = (lens - 1).clamp(min = 0)[:, None])
        random_goal = einx.get_at('b [t] d, b q -> b q d', states, random_idx)

        coin = torch.rand(batch, timesteps, device = device)
        base = base_goal.expand(batch, timesteps, GOAL_DIM)
        goal = torch.where((coin < p_base)[..., None], base,
               torch.where((coin < p_base + p_future)[..., None], future_goal,
               torch.where((coin < p_base + p_future + p_random)[..., None], random_goal, states)))

        reach_index, cutoff = compute_first_dwell_time(goal, next_states = states, eps = eps, dwell_steps = dwell, lens = lens)

        dwell_feature = torch.full((batch, timesteps, 1), dwell / DWELL_SCHEDULE[-1][1], device = device)
        event = torch.cat((goal, dwell_feature), dim = -1)

        valid = lens_to_mask(lens, timesteps)

        return tuple(t[valid] for t in (obs, action, event, reach_index, cutoff))

    # update - maximum likelihood hazard critic, then an actor maximizing the survival value through the critic

    def update(obs, action, lens, dwell, target_entropy):
        obs, action, lens = obs.to(device), action.to(device), lens.to(device)
        states, actions, events, reach_index, cutoff = relabel(obs, action, lens, dwell)

        critic_optim.zero_grad()
        critic_loss = critic(states, actions, events, reach_event_index = reach_index, horizon_cutoff = cutoff)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        critic_optim.step()

        actions_new = actor(states, events)
        values = critic(states, actions_new, events, return_values = True)
        entropy = actor.dist(states, events).entropy()

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

    # evaluation - fixed seeds so checkpoints are compared on identical episodes

    def evaluate():
        eval_stats = evaluate_actor(
            lambda state, goal: actor.logits(state, goal).argmax(dim = -1).squeeze(-1),
            eval_env,
            episodes = num_eval_episodes,
            max_steps = horizon,
            seed = seed,
            goal = base_goal,
            device = device
        )

        return eval_stats.mean

    print(f'action repr: {action_repr} | device {device}', flush = True)
    print(f'initial eval {evaluate():.1f}', flush = True)

    best_score = -float('inf')

    for iteration in range(1, iterations + 1):
        dwell = schedule_value(DWELL_SCHEDULE, iteration)
        epsilon = schedule_value(EPSILON_SCHEDULE, iteration)
        target_entropy = schedule_value(ENTROPY_SCHEDULE, iteration)

        collect(epsilon, warmup = iteration <= warmup)

        loader = buffer.dataloader(batch_size = batch_episodes, to_named_tuple = ('obs', 'action', '_lens'), shuffle = True)
        losses = [update(batch.obs, batch.action, batch.lens, dwell, target_entropy) for _, batch in zip(range(grad_steps), loader)]

        if not divisible_by(iteration, eval_every) and iteration != iterations:
            continue

        score = evaluate()

        if score > best_score:
            best_score = score
            torch.save(dict(actor = actor.state_dict(), critic = critic.state_dict(), log_alpha = log_alpha, action_repr = action_repr, iteration = iteration), checkpoint_file)

        print(f'iteration {iteration:4d} | dwell {dwell:2d} | train return {stats["collect_return"]:5.1f} | critic loss {np.mean([l[0] for l in losses]):.4f} | value {np.mean([l[1] for l in losses]):.2f} | eval return {score:5.1f}', flush = True)

        if score >= target_return:
            break

    for env in (*envs, eval_env):
        env.close()
    temp_dir.cleanup()

    assert score >= target_return, f'expected cartpole to balance, final eval return: {score:.1f}'
    print(f'cartpole balanced with survival RL ({action_repr} actions, eval return: {score:.1f})')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
