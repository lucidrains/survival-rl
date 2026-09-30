#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "einx",
#     "einops",
#     "fire",
#     "imageio",
#     "imageio-ffmpeg",
#     "mean-conc-beta",
#     "numpy",
#     "survival-rl",
#     "torch>=2.5",
#     "torch-einops-utils>=0.1.31",
#     "x-mlps-pytorch>=0.6.4",
# ]
# ///

# train_hazard_lander - 1d powered descent
# goal event     -> soft touchdown (y ~ 0, vy ~ 0, contact = 1)
# bad goal event -> hard impact   (y ~ 0, |vy| > v_safe, contact = 1)
# actor maximizes time-to-goal value, and for beta > 0 also minimizes time-to-bad-goal value

from __future__ import annotations
from math import log
from pathlib import Path
from collections import deque
import random

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import einx
from einops import rearrange, repeat

from mean_conc_beta import Beta
from survival_rl import HazardCriticCompetitive, compute_first_dwell_time
from torch_einops_utils import lens_to_mask
from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else (d() if callable(d) else d)

def divisible_by(num, den):
    return (num % den) == 0

def schedule_value(schedule, step):
    for until, value in schedule:
        if step <= until:
            return value
    return schedule[-1][1]

def as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else (v.replace(',', ' ').split() if isinstance(v, str) else [v])

# env - y, vy, contact, first order engine lag and thrust noise

class HazardLander:
    def __init__(self, num_envs, horizon = 100, dt = 0.05, gravity = 0.5, thrust = 2., engine_tau = 0.1, v_safe = 0.5, thrust_noise = 0.1, y_range = (0.3, 0.5), vy_range = (-0.2, 0.1), device = None):
        self.num_envs, self.horizon, self.dt, self.gravity, self.thrust = num_envs, horizon, dt, gravity, thrust
        self.engine_tau, self.v_safe, self.thrust_noise = engine_tau, v_safe, thrust_noise
        self.y_range, self.vy_range = y_range, vy_range
        self.device, self.noise_scale = device, 1.
        self.reset()

    def reset(self, mask = None):
        idx = slice(None) if not exists(mask) else mask
        n = self.num_envs if not exists(mask) else int(mask.sum())
        if not exists(mask):
            self.engine = torch.zeros(self.num_envs, device = self.device)
            self.t = torch.zeros(self.num_envs, dtype = torch.long, device = self.device)
            self.y, self.vy = torch.empty(self.num_envs, device = self.device), torch.empty(self.num_envs, device = self.device)
        else:
            self.engine[mask] = 0.
            self.t[mask] = 0

        if n > 0:
            self.y[idx] = torch.empty(n, device = self.device).uniform_(*self.y_range)
            self.vy[idx] = torch.empty(n, device = self.device).uniform_(*self.vy_range)

        return self.obs()

    def obs(self):
        return torch.stack((self.y.clamp(min = 0.), self.vy, (self.y <= 0.).to(self.y.dtype)), dim = -1)

    def step(self, action):
        u = (action[..., 0] + self.noise_scale * self.thrust_noise * torch.randn_like(action[..., 0])).clamp(-1., 1.)
        self.engine = self.engine + (u - self.engine) * (self.dt / self.engine_tau)
        self.vy = self.vy + (self.engine * self.thrust - self.gravity) * self.dt
        self.y = self.y + self.vy * self.dt
        self.t += 1

        contact = self.y <= 0.
        success, crash = contact & (self.vy.abs() <= self.v_safe), contact & (self.vy.abs() > self.v_safe)
        done = success | crash | (self.t >= self.horizon)
        return self.obs(), torch.where(success, 1, torch.where(crash, 2, 0)).long(), done

# actor

class Actor(nn.Module):
    def __init__(self, dim_state = 3, dim_event = 3, dim_action = 1, dim_hidden = 64, init_conc = 4.):
        super().__init__()
        self.dim_event, self.dim_action = dim_event, dim_action
        self.net = MLP(dim_state + dim_event, dim_hidden, dim_hidden, 2 * dim_action, activation = nn.SiLU())
        self.beta = Beta(bounds = (-1., 1.), init_conc = init_conc, detach_entropy_mean = True)

    def dist(self, state, event):
        logits = self.net(torch.cat((state, event[..., :self.dim_event]), dim = -1))
        return self.beta(rearrange(logits, '... (d p) -> ... d p', d = self.dim_action))

# collection

def collect(actor, env, buffer, goal, epsilon, num_episodes):
    obs, trajs = env.reset(), [[] for _ in range(env.num_envs)]
    counts, total_len, finished = [0, 0, 0], 0, 0

    while finished < num_episodes:
        with torch.no_grad():
            actions = actor.dist(obs, repeat(goal, 'd -> b d', b = obs.shape[0])).sample()
            is_random = rearrange(torch.rand(obs.shape[0], device = obs.device) < epsilon, 'b -> b 1')
            actions = torch.where(is_random, torch.empty_like(actions).uniform_(-1., 1.), actions)

        prev_obs, obs, outcome, done = obs, *env.step(actions)
        reset_mask = torch.zeros(env.num_envs, dtype = torch.bool, device = obs.device)

        for i in range(env.num_envs):
            trajs[i].append((prev_obs[i], actions[i], obs[i]))

            if bool(done[i]) and finished < num_episodes:
                t = trajs[i]
                buffer.append(dict(
                    obs = torch.stack([x[0] for x in t]),
                    action = torch.stack([x[1] for x in t]),
                    next_obs = torch.stack([x[2] for x in t]),
                    outcome = int(outcome[i])
                ))
                counts[int(outcome[i])] += 1
                total_len += len(t)
                trajs[i] = []
                reset_mask[i] = True
                finished += 1

        if bool(reset_mask.any()):
            env.reset(mask = reset_mask)

    return dict(timeout = counts[0], success = counts[1], crash = counts[2], mean_len = total_len / max(finished, 1))

def make_batch(episodes, device):
    batch, length = len(episodes), max(len(e['action']) for e in episodes)
    obs, action, next_obs = torch.zeros(batch, length, 3, device = device), torch.zeros(batch, length, 1, device = device), torch.zeros(batch, length, 3, device = device)
    lens, outcome = torch.tensor([len(e['action']) for e in episodes], device = device), torch.tensor([e['outcome'] for e in episodes], device = device)

    for i, e in enumerate(episodes):
        n = len(e['action'])
        obs[i, :n], action[i, :n], next_obs[i, :n] = e['obs'], e['action'], e['next_obs']

    return obs, action, next_obs, lens, outcome

# update

def update(actor, critic_goal, critic_bad, optims, log_alpha, batch, cfg, beta):
    device = cfg['goal'].device
    actor_optim, goal_optim, bad_optim, alpha_optim = optims
    obs, action, next_obs, lens, outcome = batch

    batch_size, timesteps = obs.shape[:2]
    times = torch.arange(timesteps, device = device)
    valid = lens_to_mask(lens, timesteps)

    # goal relabeling - base / trajectory / random goals

    span = (lens[:, None] - times).clamp(min = 1)
    future_idx = torch.minimum(times + (torch.rand_like(span.float()) * span).long(), (lens - 1)[:, None])
    traj_goal = einx.get_at('b [t] d, b q -> b q d', next_obs, future_idx)

    rand_idx = torch.minimum((torch.rand(batch_size, timesteps, device = device) * lens[:, None]).long(), (lens - 1)[:, None])
    rand_goal = einx.get_at('b [t] d, b q -> b q d', next_obs, rand_idx)

    coin = torch.rand(batch_size, timesteps, device = device)
    is_base = coin < cfg['p_base']
    is_traj = (coin >= cfg['p_base']) & (coin < cfg['p_base'] + cfg['p_trajectory'])

    is_base_mask, is_traj_mask = rearrange(is_base, 'b t -> b t 1'), rearrange(is_traj, 'b t -> b t 1')
    goal = torch.where(is_base_mask, cfg['goal'], torch.where(is_traj_mask, traj_goal, rand_goal))

    reach, cutoff = compute_first_dwell_time(goal, next_states = next_obs, eps = cfg['eps'], dwell_steps = cfg['dwell'], lens = lens)

    # unreached base goal is censored at the full horizon by default

    if cfg['goal_censor'] == 'horizon':
        cutoff = torch.where(is_base & (reach < 0), cfg['horizon'] - times, cutoff)

    # undesirable goals - hard ground impact, crashing episodes query their own terminal state

    crash = outcome == 2
    terminal = einx.get_at('b [t] d, b -> b d', next_obs, (lens - 1).clamp(min = 0))

    vy_bad = -(cfg['v_safe'] + cfg['bad_margin'] + torch.rand(batch_size, device = device) * (cfg['bad_vy_max'] - cfg['v_safe'] - cfg['bad_margin']))
    sampled_bad = torch.stack((torch.zeros_like(vy_bad), vy_bad, torch.ones_like(vy_bad)), dim = -1)

    bad_goal_per_batch = torch.where(rearrange(crash, 'b -> b 1'), terminal, sampled_bad)
    bad_goal = repeat(bad_goal_per_batch, 'b d -> b t d', t = timesteps)

    remaining = rearrange(lens - 1, 'b -> b 1') - times
    bad_reach = torch.where(rearrange(crash, 'b -> b 1'), remaining, -1)
    bad_cutoff = rearrange(lens, 'b -> b 1') - times

    states, actions = obs[valid], action[valid]
    goal_ev, bad_goal = goal[valid], bad_goal[valid]
    reach, cutoff = reach[valid], cutoff[valid]
    bad_reach, bad_cutoff = bad_reach[valid], bad_cutoff[valid]

    # goal hazard critic

    loss_goal = critic_goal(states, actions, goal_ev, reach_event_index = reach, horizon_cutoff = cutoff)
    loss_goal.backward()
    nn.utils.clip_grad_norm_(critic_goal.parameters(), cfg['max_grad_norm'])
    goal_optim.step()
    goal_optim.zero_grad()

    # hazard critic for bad events

    loss_bad = critic_bad(states, actions, bad_goal, reach_event_index = bad_reach, horizon_cutoff = bad_cutoff)
    loss_bad.backward()
    nn.utils.clip_grad_norm_(critic_bad.parameters(), cfg['max_grad_norm'])
    bad_optim.step()
    bad_optim.zero_grad()

    # actor - maximize time-to-goal value, minimize time-to-bad-goal value

    dist = actor.dist(states, goal_ev)
    actions_flat = rearrange(dist.rsample(), 'b ... -> b (...)')
    v_goal = critic_goal(states, actions_flat, goal_ev, return_values = True)
    v_bad = critic_bad(states, actions_flat, bad_goal, return_values = True)
    entropy = rearrange(dist.entropy(), 'b ... -> b (...)').sum(dim = -1)

    alpha = log_alpha.exp()
    actor_loss = (-alpha.detach() * entropy - v_goal).mean() + beta * v_bad.mean()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg['max_grad_norm'])
    actor_optim.step()
    actor_optim.zero_grad()

    alpha_loss = (alpha * (entropy.detach() - cfg['target_entropy'])).mean()
    alpha_loss.backward()
    alpha_optim.step()
    alpha_optim.zero_grad()

    log_alpha.data.clamp_(min = log(cfg['min_alpha']), max = log(cfg['max_alpha']))

    return dict(
        loss_goal = loss_goal.item(),
        loss_bad = loss_bad.item(),
        v_goal = v_goal.mean().item(),
        v_bad = v_bad.mean().item(),
        entropy = entropy.mean().item(),
        alpha = alpha.item()
    )

# evaluation

@torch.no_grad()
def evaluate(actor, env, goal, noise_scale = 1.):
    env.noise_scale = noise_scale
    obs = env.reset()
    outcomes = torch.full((env.num_envs,), -1, dtype = torch.long, device = obs.device)
    alive = torch.ones(env.num_envs, dtype = torch.bool, device = obs.device)

    while bool(alive.any()):
        obs, outcome, done = env.step(actor.dist(obs, repeat(goal, 'd -> b d', b = obs.shape[0])).mean)
        newly = done & alive
        outcomes[newly] = outcome[newly]
        alive = alive & ~done

    env.noise_scale = 1.
    return dict(
        success = (outcomes == 1).float().mean().item(),
        crash = (outcomes == 2).float().mean().item(),
        timeout = (outcomes == 0).float().mean().item()
    )

# rendering

def render_frame(y, vy, u, v_safe):
    height, width, ground = 240, 240, int(240 * 0.85)
    frame = np.empty((height, width, 3), dtype = np.uint8)
    frame[:], frame[ground:] = (135, 206, 235), (70, 70, 70)
    frame[(ground - 108):(ground - 106), :8] = 255

    px, py = width // 2, min(max(int(ground - (min(max(y, 0.), 1.4) / 1.4) * ground), 8), ground - 4)
    frame[py - 6:py + 6, px - 12:px + 12] = (220, 70, 70) if abs(vy) > v_safe else (70, 190, 90)

    if u > 0.05:
        frame[py + 6:py + 6 + int(4 + 16 * u), px - 5:px + 5] = (255, 150, 40)

    bx, fill = width - 20, int(min(abs(vy) / 1.5, 1.) * (ground - 30))
    frame[ground - 20 - fill:ground - 20, bx:bx + 12] = (200, 60, 60) if abs(vy) > v_safe else (60, 120, 200)
    return frame

def record_episode(actor, cfg, path, device):
    env = HazardLander(1, horizon = cfg['horizon'], v_safe = cfg['v_safe'], thrust = cfg['thrust'], thrust_noise = cfg['thrust_noise'], device = device)
    obs, frames, done = env.reset(), [], torch.zeros(1, dtype = torch.bool, device = device)

    while not bool(done.all()):
        with torch.no_grad():
            action = actor.dist(obs, repeat(cfg['goal'], 'd -> 1 d')).mean
        frames.append(render_frame(float(obs[0, 0]), float(obs[0, 1]), float(action[0, 0]), cfg['v_safe']))
        obs, _, done = env.step(action)

    path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(path), frames, fps = 20)

# training

def train(seed, beta, cfg, device):
    torch.manual_seed(seed)
    random.seed(seed)

    env = HazardLander(cfg['num_envs'], horizon = cfg['horizon'], v_safe = cfg['v_safe'], thrust = cfg['thrust'], thrust_noise = cfg['thrust_noise'], device = device)
    eval_env = HazardLander(cfg['eval_episodes'], horizon = cfg['horizon'], v_safe = cfg['v_safe'], thrust = cfg['thrust'], thrust_noise = cfg['thrust_noise'], device = device)

    actor = Actor(dim_hidden = cfg['dim_hidden']).to(device)

    critic_kwargs = dict(dim = cfg['dim'], depth = cfg['depth'], dim_state = 3, num_actions = 1, pred_time_bins = cfg['horizon'], dim_event = 3, discount_factor = cfg['gamma'])
    critic_goal, critic_bad = (HazardCriticCompetitive(**critic_kwargs).to(device) for _ in range(2))

    log_alpha = nn.Parameter(torch.tensor(log(cfg['init_alpha']), device = device))
    optims = (
        AdamW(actor.parameters(), lr = cfg['lr_actor']),
        AdamW(critic_goal.parameters(), lr = cfg['lr_critic']),
        AdamW(critic_bad.parameters(), lr = cfg['lr_critic']),
        AdamW([log_alpha], lr = cfg['lr_actor'])
    )

    buffer, records = deque(maxlen = cfg['buffer_size']), []

    for iteration in range(1, cfg['iterations'] + 1):
        col = collect(actor, env, buffer, cfg['goal'], schedule_value(cfg['epsilon_schedule'], iteration), cfg['num_envs'])

        for _ in range(cfg['sgd_steps']):
            batch = make_batch(random.sample(buffer, min(cfg['batch_episodes'], len(buffer))), device)
            stats = update(actor, critic_goal, critic_bad, optims, log_alpha, batch, cfg, beta)

        if not (divisible_by(iteration, cfg['eval_every']) or iteration == cfg['iterations']):
            continue

        ev = evaluate(actor, eval_env, cfg['goal'])
        stress = evaluate(actor, eval_env, cfg['goal'], noise_scale = cfg['stress_noise'])

        if cfg['record_video']:
            record_episode(actor, cfg, Path(cfg['video_dir']) / f'{cfg["goal_censor"]}_beta{beta:.2f}_seed{seed}_iter{iteration:04d}.mp4', device)

        print(f'beta {beta:4.2f} | seed {seed} | iter {iteration:4d} | succ {ev["success"]:.2f} | crash {ev["crash"]:.2f} | t/o {ev["timeout"]:.2f} | stress crash {stress["crash"]:.2f} | vg {stats["v_goal"]:7.1f} | vb {stats["v_bad"]:7.1f} | coll s{col["success"]} c{col["crash"]} t{col["timeout"]} len {col["mean_len"]:.0f}', flush = True)

        records.append(dict(
            beta = beta, seed = seed, iteration = iteration,
            success = ev['success'], crash = ev['crash'], timeout = ev['timeout'],
            stress_success = stress['success'], stress_crash = stress['crash'], stress_timeout = stress['timeout']
        ))

    return records

# main

def main(
    seeds = '0', betas = '0.0 0.1 0.5',
    iterations = 150, num_envs = 32, eval_episodes = 128, eval_every = 15,
    buffer_size = 2048, batch_episodes = 8, sgd_steps = 8,
    dim = 64, depth = 2, dim_hidden = 64, gamma = 0.99,
    horizon = 100, eps = 0.5, dwell = 1,
    lr_actor = 3e-4, lr_critic = 1e-3, init_alpha = 0.05, min_alpha = 0.001, max_alpha = 0.2,
    target_entropy = -0.5, max_grad_norm = 1.,
    p_base = 0.4, p_trajectory = 0.4,
    v_safe = 0.5, thrust = 1., thrust_noise = 0.3,
    goal_censor = 'horizon', bad_margin = 0.1, bad_vy_max = 1.5, stress_noise = 3.,
    epsilon_schedule = ((1, 1.0), (30, 0.3), (80, 0.1), (150, 0.05)),
    video_dir = 'videos_hazard', record_video = True,
    out = 'hazard_lander_results.csv', device = ''
):
    device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    cfg = {**locals(), 'goal': torch.tensor([0., 0., 1.], device = device)}
    print(f'device {device}', flush = True)

    rows = []
    for seed in (int(s) for s in as_list(seeds)):
        for beta in (float(b) for b in as_list(betas)):
            print(f'--- seed {seed} beta {beta} ---', flush = True)
            rows.extend(train(seed, beta, cfg, device))

    # final eval per arm -> mean +/- std over seeds

    keys = ('success', 'crash', 'timeout', 'stress_success', 'stress_crash')
    print('\n=== final eval (last iteration per run, mean +/- std over seeds) ===\n' + f'{"beta":>5} | ' + ' | '.join(f'{k:>13}' for k in keys), flush = True)

    for beta in sorted({r['beta'] for r in rows}):
        last = {r['seed']: r for r in rows if r['beta'] == beta}
        line = [f'{beta:5.2f}'] + [f'{torch.tensor([r[k] for r in last.values()]).mean():6.3f} +/- {(torch.tensor([r[k] for r in last.values()]).std().item() if len(last) > 1 else 0.):.3f}' for k in keys]
        print(' | '.join(line), flush = True)

    out = Path(out)
    out.parent.mkdir(parents = True, exist_ok = True)
    header = ['beta', 'seed', 'iteration', *keys, 'stress_timeout']
    out.write_text(','.join(header) + '\n' + ''.join(','.join(str(r[k]) for k in header) + '\n' for r in rows))
    print(f'wrote {out}', flush = True)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
