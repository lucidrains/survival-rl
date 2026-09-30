#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "env-ssl-wrapper>=0.5.0",
#     "fire",
#     "gymnasium[box2d]",
#     "imageio",
#     "imageio-ffmpeg",
#     "mean-conc-beta",
#     "memmap-replay-buffer",
#     "numpy",
#     "survival-rl",
# ]
# ///

from __future__ import annotations
from math import log
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import gymnasium as gym
import einx
from einops import rearrange, repeat

from env_ssl_wrapper import ActionChunkWrapper, evaluate_actor
from mean_conc_beta import Beta
from memmap_replay_buffer import ReplayBuffer
from survival_rl import HazardCriticCompetitive, compute_first_dwell_time
from torch_einops_utils import cast_tensor, lens_to_mask, tree_map_tensor_to_device
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

# state pool for visited states

class StatePool:
    def __init__(self, dim, max_size = 4096, device = None):
        self.data = torch.zeros(max_size, dim, device = device)
        self.ptr = self.size = 0

    def add(self, x):
        x = rearrange(cast_tensor(x, device = self.data.device)[..., :self.data.shape[-1]], '... d -> (...) d')
        n = min(len(x), len(self.data))
        idx = (self.ptr + torch.arange(n, device = self.data.device)) % len(self.data)
        self.data[idx] = x[-n:]
        self.ptr = (self.ptr + n) % len(self.data)
        self.size = min(self.size + n, len(self.data))

    def sample(self, n):
        if self.size == 0:
            return self.data.new_zeros(n, self.data.shape[-1])
        return self.data[torch.randint(self.size, (n,), device = self.data.device)]

# actor

class Actor(nn.Module):
    def __init__(self, dim_state, dim_goal, dim_action, dim_hidden = 256, chunk_len = 2, init_conc = 4.):
        super().__init__()
        self.dim_goal, self.chunk_len, self.dim_action = dim_goal, chunk_len, dim_action
        self.net = MLP(dim_state + dim_goal, dim_hidden, dim_hidden, 2 * dim_action * chunk_len, activation = nn.SiLU())
        self.beta = Beta(bounds = (-1., 1.), init_conc = init_conc, detach_entropy_mean = True)

    def dist(self, state, goal):
        x = torch.cat((state, goal[..., :self.dim_goal]), dim = -1)
        logits = rearrange(self.net(x), '... (c d p) -> ... c d p', c = self.chunk_len, d = self.dim_action, p = 2)
        return self.beta(logits)

# main

def main(
    seed = 0,
    horizon = 500,
    num_envs = 16,
    batch_episodes = 8,
    replay_buffer_size = 256,
    state_pool_size = 4096,
    dim_state = 8, dim_goal = 6, dim_action = 2,
    chunk_len = 2,
    dim_hidden = 256, depth = 2,
    init_conc = 4.,
    init_alpha = 0.02, min_alpha = 0.001, max_alpha = 0.2,
    max_grad_norm = 1.,
    grad_steps = 16,
    iterations = 3500,
    eval_every = 25, video_every = 50, checkpoint_every = 100,
    warmup = 3,
    target_return = 20., landed_score_threshold = 50.,
    trajectory_dwell = 1, k_max = 16,
    eps = 0.35, gamma = 0.99,
    lr_actor = 3e-4, lr_critic = 1e-3,
    p_base = 0.40, p_trajectory = 0.40, p_random = 0.10, collect_base_prob = 0.8,
    num_eval_episodes = 10,
    dwell_schedule = ((60, 1), (150, 2), (350, 3), (700, 4), (1100, 6), (1600, 8), (2300, 10), (3500, 12)),
    epsilon_schedule = ((100, 0.25), (400, 0.1), (1500, 0.05), (3500, 0.02)),
    entropy_schedule = ((300, -1.0), (800, -1.5), (2000, -2.0), (3500, -2.5)),
    video_dir = 'videos',
    record_video = True,
    resume = False,
    checkpoint_file = 'lunar_checkpoint.pt',
    device = ''
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))

    assert divisible_by(horizon, chunk_len)
    chunk_horizon = horizon // chunk_len
    gamma_chunk = gamma ** chunk_len

    actor = Actor(dim_state, dim_goal, dim_action, dim_hidden, chunk_len, init_conc).to(device)

    critic = HazardCriticCompetitive(
        dim = dim_hidden,
        depth = depth,
        dim_state = dim_state,
        num_actions = dim_action * chunk_len,
        pred_time_bins = chunk_horizon,
        dim_event = dim_goal + 1,
        discount_factor = gamma_chunk
    ).to(device)

    actor_optim, critic_optim = AdamW(actor.parameters(), lr = lr_actor), AdamW(critic.parameters(), lr = lr_critic)

    log_alpha = nn.Parameter(torch.tensor(log(init_alpha), device = device))
    alpha_optim = AdamW([log_alpha], lr = lr_actor)

    temp_dir = tempfile.TemporaryDirectory()
    buffer = ReplayBuffer(
        temp_dir.name,
        max_episodes = replay_buffer_size,
        max_timesteps = chunk_horizon,
        fields = dict(obs = ('float', dim_state), action = ('float', dim_action * chunk_len), next_obs = ('float', dim_state)),
        circular = True,
        overwrite = True
    )

    envs = [ActionChunkWrapper(gym.make('LunarLanderContinuous-v3'), chunk_len = chunk_len, reward_mode = 'chunk') for _ in range(num_envs)]
    eval_env = ActionChunkWrapper(gym.make('LunarLanderContinuous-v3', render_mode = 'rgb_array'), chunk_len = chunk_len, reward_mode = 'chunk')

    base_goal = torch.zeros(dim_goal, device = device)
    state_pool = StatePool(dim = dim_goal, max_size = state_pool_size, device = device)
    stats = dict()

    # evaluation

    def evaluate(video_path = None, episodes = num_eval_episodes):
        return evaluate_actor(actor, eval_env, episodes = episodes, max_steps = chunk_horizon, seed = seed, goal = base_goal, video_path = video_path).mean

    # collect episodes

    def collect(epsilon, warmup = False):
        obs = [env.reset()[0] for env in envs]
        state_pool.add(np.stack(obs))

        trajs = [dict(obs = [], action = [], next_obs = [], score = 0., run = 0, dwell = 0) for _ in range(num_envs)]
        active = list(range(num_envs))
        lengths, scores, landed = [], [], 0

        is_base = rearrange(torch.rand(num_envs, device = device) < collect_base_prob, 'b -> b 1')
        env_goals = torch.where(is_base, base_goal, state_pool.sample(num_envs))

        while active:
            batch = np.stack([obs[i] for i in active])
            states = cast_tensor(batch, device = device)
            state_pool.add(states)

            in_goal = (states[..., :dim_goal] - base_goal).norm(dim = -1) <= eps

            if warmup:
                actions = torch.empty(len(active), chunk_len, dim_action, device = device).uniform_(-1., 1.)
            else:
                with torch.no_grad():
                    actions = actor.dist(states, env_goals[active]).sample()
                    is_random = rearrange(torch.rand(len(active), device = device) < epsilon, 'b -> b 1 1')
                    actions = torch.where(is_random, torch.empty_like(actions).uniform_(-1., 1.), actions)

            in_goal_list = in_goal.tolist()
            actions_np = actions.cpu().numpy()

            for j, i in enumerate(list(active)):
                t, act = trajs[i], actions_np[j]
                t['run'] = t['run'] + 1 if in_goal_list[j] else 0
                t['dwell'] = max(t['dwell'], t['run'])

                next_o, reward, terminated, truncated, info = envs[i].step(act[None])
                chunk_r = float(np.sum(info.get('chunk_rewards', [reward])))

                t['obs'].append(batch[j])
                t['action'].append(rearrange(act, 'c d -> (c d)'))
                t['next_obs'].append(np.asarray(next_o, dtype = np.float32))
                t['score'] += chunk_r
                obs[i] = next_o

                if terminated or truncated or len(t['obs']) >= chunk_horizon:
                    buffer.store_episode(obs = np.stack(t['obs']), action = np.stack(t['action']), next_obs = np.stack(t['next_obs']))
                    landed += int(terminated and t['score'] > landed_score_threshold)
                    lengths.append(len(t['obs']))
                    scores.append(t['score'])
                    active.remove(i)

        stats['collect_return'] = float(np.mean(scores))
        stats['collect_len'] = float(np.mean(lengths))
        stats['landed'] = landed
        stats['achieved'] = float(np.mean([t['dwell'] for t in trajs]))

    # relabel

    def relabel(obs, action, next_obs, lens, dwell):
        batch, timesteps, _ = obs.shape
        lens = lens.long()
        times = torch.arange(timesteps, device = device)
        positions = obs[..., :dim_goal]
        next_positions = next_obs[..., :dim_goal]

        time_delta = rearrange(times, 'i -> 1 i') - rearrange(times, 'j -> j 1')
        weights = (gamma_chunk ** time_delta.clamp(min = 0.)).masked_fill(time_delta < 1, 0.)
        weights = weights + (weights.sum(-1, keepdim = True) <= 0.) / timesteps
        future_idx = torch.multinomial(weights, 1)[:, 0]
        future_idx = repeat(future_idx, 't -> b t', b = batch).clamp(max = (lens - 1).clamp(min = 0)[:, None])

        trajectory_goal = einx.get_at('b [t] d, b q -> b q d', next_positions, future_idx)

        valid = lens_to_mask(lens, timesteps)

        random_idx = torch.randint(timesteps, (batch, timesteps), device = device).clamp(max = (lens - 1).clamp(min = 0)[:, None])
        random_goal = einx.get_at('b [t] d, b q -> b q d', positions, random_idx)

        coin = torch.rand(batch, timesteps, device = device)
        is_base = coin < p_base
        is_trajectory = (coin >= p_base) & (coin < p_base + p_trajectory)
        is_random = (coin >= p_base + p_trajectory) & (coin < p_base + p_trajectory + p_random)

        is_base_mask, is_traj_mask, is_rand_mask = (rearrange(c, 'b t -> b t 1') for c in (is_base, is_trajectory, is_random))

        goal = torch.where(is_base_mask, base_goal,
               torch.where(is_traj_mask, trajectory_goal,
               torch.where(is_rand_mask, random_goal, positions)))

        dwell_steps = torch.where(is_base, dwell, trajectory_dwell)

        reach_index, cutoff = compute_first_dwell_time(goal, next_states = next_positions, eps = eps, dwell_steps = dwell_steps, lens = lens)
        cutoff = torch.where(is_base & (reach_index < 0), chunk_horizon - times, cutoff)

        reached = reach_index >= 0
        stats['event_rate'] = (reached & valid).float().sum().item() / valid.sum().clamp(min = 1).item()
        stats['hit_time'] = reach_index[reached & valid].float().mean().item() if (reached & valid).any() else -1.

        dwell_feature = torch.where(is_base_mask, dwell / k_max, trajectory_dwell / k_max)
        event = torch.cat((goal, dwell_feature), dim = -1)

        return tuple(t[valid] for t in (obs, action, event, reach_index, cutoff))

    # update

    def update(obs, action, next_obs, lens, dwell, target_entropy):
        obs, action, next_obs, lens = tree_map_tensor_to_device((obs, action, next_obs, lens), device)
        states, actions, events, reach_index, cutoff = relabel(obs, action, next_obs, lens, dwell)

        critic_loss = critic(states, actions, events, reach_event_index = reach_index, horizon_cutoff = cutoff)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optim.step()
        critic_optim.zero_grad()

        dist = actor.dist(states, events)
        actions_flat = rearrange(dist.rsample(), 'b ... -> b (...)')
        values = critic(states, actions_flat, events, return_values = True)
        entropy = dist.entropy().sum(dim = (-1, -2))

        alpha = log_alpha.exp()

        actor_loss = (-alpha.detach() * entropy - values).mean()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()
        actor_optim.zero_grad()

        alpha_loss = (alpha * (entropy.detach() - target_entropy)).mean()
        alpha_loss.backward()
        alpha_optim.step()
        alpha_optim.zero_grad()

        log_alpha.data.clamp_(min = log(min_alpha), max = log(max_alpha))

        stats['alpha'] = alpha.item()
        stats['entropy'] = entropy.mean().item()

        return critic_loss.item(), values.mean().item()

    start_iteration, best_score = 1, -float('inf')
    video_dir, ckpt_path = Path(video_dir), Path(checkpoint_file)
    video_dir.mkdir(parents = True, exist_ok = True)

    def save_checkpoint(path):
        torch.save(dict(actor = actor.state_dict(), critic = critic.state_dict(), log_alpha = log_alpha, iteration = iteration, best_score = best_score), path)

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location = device)
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        log_alpha.data.copy_(ckpt['log_alpha'].data)
        start_iteration, best_score = ckpt.get('iteration', 0) + 1, ckpt.get('best_score', -float('inf'))
        print(f'resumed from {checkpoint_file} at {start_iteration} (best: {best_score:.1f})')

    score = evaluate()
    print(f'initial eval {score:.1f}', flush = True)

    for iteration in range(start_iteration, iterations + 1):
        dwell = schedule_value(dwell_schedule, iteration)
        epsilon = schedule_value(epsilon_schedule, iteration)
        target_entropy = schedule_value(entropy_schedule, iteration) * chunk_len

        collect(epsilon, warmup = iteration <= warmup)

        loader = buffer.dataloader(batch_size = batch_episodes, to_named_tuple = ('obs', 'action', 'next_obs', '_lens'), shuffle = True)
        losses = [update(batch.obs, batch.action, batch.next_obs, batch.lens, dwell, target_entropy) for _, batch in zip(range(grad_steps), loader)]

        if divisible_by(iteration, eval_every) or iteration == iterations:
            save_video = record_video and (divisible_by(iteration, video_every) or iteration == iterations)
            video_path = f'{video_dir}/eval_iter_{iteration:04d}.mp4' if save_video else None
            score = evaluate(video_path = video_path)

            critic_loss, mean_val = np.mean(losses, axis = 0)
            print(f'iteration {iteration:4d} | dwell {dwell:2d} | achieved {stats["achieved"]:.1f} | critic {critic_loss:.4f} | val {mean_val:.2f} | alpha {stats["alpha"]:.3f} | ent {stats["entropy"]:.2f} | coll ret {stats["collect_return"]:.1f} | len {stats["collect_len"]:.0f} | landed {stats["landed"]:2d} | ev {stats["event_rate"]:.3f} | hit {stats["hit_time"]:.1f} | eval return {score:.1f}', flush = True)

            if score > best_score:
                best_score = score
                save_checkpoint('lunar_best.pt')
                if record_video:
                    evaluate(video_path = f'{video_dir}/best_landing.mp4', episodes = 1)

            if divisible_by(iteration, checkpoint_every):
                save_checkpoint(checkpoint_file)

            if score >= target_return:
                break

    for env in (*envs, eval_env):
        env.close()
    temp_dir.cleanup()

    assert score >= target_return, f'expected lunar lander to land, final eval return: {score:.1f}'
    print(f'lunar lander landed with survival RL (eval return: {score:.1f})')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
