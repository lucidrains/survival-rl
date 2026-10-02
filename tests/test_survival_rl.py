import pytest
param = pytest.mark.parametrize

import torch
from torch.nn import Module, Linear

from x_mlps_pytorch import MLP
from survival_rl import HazardCritic, HazardCriticCompetitive, compute_first_dwell_time

# mock actors

class Actor(Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(8, 16, 16, 7)

    def forward(self, state, event):
        return self.mlp((state, event))

class CondActor(Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(8, 16, 16, 7)

    def forward(self, state, cond):
        return self.mlp((state, cond))

# helpers

def get_log_survival(output):
    return output.log_survival if isinstance(output, tuple) else output

# test

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
@param('reach_event_index, horizon_cutoff, shape', [
    (None, None, (2, 4)),
    (torch.tensor([0, 1]), None, ()),
    (torch.tensor([1, 2]), torch.tensor([3, 2]), ()),
])
def test_critic(critic_klass, reach_event_index, horizon_cutoff, shape):
    critic = critic_klass(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    output = critic(state, action, event, reach_event_index = reach_event_index, horizon_cutoff = horizon_cutoff)

    if isinstance(output, tuple):
        num_logits = 5 if critic_klass is HazardCriticCompetitive else 4
        assert output.logits.shape == (2, num_logits)
        output = output.log_survival

    assert output.shape == shape

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
@param('encode_event', (True, False))
def test_overridable_event_encoder(critic_klass, encode_event):
    torch.manual_seed(0)

    class CustomEventEncoder(Module):
        def __init__(self, dim_event, dim_out):
            super().__init__()
            self.proj = Linear(dim_event, dim_out)

        def forward(self, event):
            return self.proj(event).relu()

    critic = critic_klass(
        dim = 16,
        depth = 2,
        dim_state = 4,
        num_actions = 7,
        pred_time_bins = 4,
        dim_event = 6,
        dim_event_encoded = 8,
        encode_event = encode_event,
        event_encoder = CustomEventEncoder(6, 8)
    )

    assert isinstance(critic.mlp.cond_encoder, CustomEventEncoder)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 6)

    output = critic(state, action, event)

    assert output.log_survival.shape == (2, 4)

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
def test_negative_reach_index_is_censored(critic_klass):
    torch.manual_seed(0)

    critic = critic_klass(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, 1]))
    censored = critic(state, action, event, reach_event_index = torch.tensor([4, 1]))

    assert torch.allclose(loss, censored)

    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, 1]), horizon_cutoff = torch.tensor([2, 4]))
    censored = critic(state, action, event, reach_event_index = torch.tensor([2, 1]), horizon_cutoff = torch.tensor([2, 4]))

    assert torch.allclose(loss, censored)

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
def test_e2e(critic_klass):
    torch.manual_seed(0)

    critic = critic_klass(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4, discount_factor = 0.9)
    optim = torch.optim.Adam(critic.parameters(), lr = 1e-2)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    reach_event_index = torch.tensor([1, 2])

    for _ in range(600):
        optim.zero_grad()
        loss = critic(state, action, event, reach_event_index = reach_event_index)
        loss.backward()
        optim.step()

    values = critic(state, action, event, return_values = True)
    expected = -torch.tensor([1.0, 1.9])

    assert torch.allclose(values, expected, atol = 1e-2)

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
def test_time_bins_are_normalized(critic_klass):
    torch.manual_seed(0)

    num_time_bins = 4

    critic = critic_klass(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = num_time_bins)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    survival = get_log_survival(critic(state, action, event)).exp()

    # bucket probabilities are survival differences, with the final bucket carrying the remaining mass

    probs = torch.cat((1. - survival[..., :1], survival[..., :-1] - survival[..., 1:], survival[..., -1:]), dim = -1)

    assert (probs >= -1e-6).all()
    assert torch.allclose(probs.sum(dim = -1), torch.ones(2), atol = 1e-5)

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
def test_censored_loss_is_tail_cross_entropy(critic_klass):
    torch.manual_seed(0)

    num_time_bins = 4

    critic = critic_klass(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = num_time_bins)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    survival = get_log_survival(critic(state, action, event)).exp()
    probs = torch.cat((1. - survival[..., :1], survival[..., :-1] - survival[..., 1:], survival[..., -1:]), dim = -1)

    # censoring at an arbitrary cutoff is the cross entropy against all remaining buckets

    cutoff = torch.tensor([2, 3])
    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, -1]), horizon_cutoff = cutoff)

    tail = torch.stack((probs[0, 2:].sum(), probs[1, 3:].sum()))
    expected = -tail.log().mean()

    assert torch.allclose(loss, expected, atol = 1e-5)

    # censoring at the horizon reduces to the final bucket cross entropy

    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, -1]))
    expected = -probs[:, num_time_bins].log().mean()

    assert torch.allclose(loss, expected, atol = 1e-5)

@param('critic_klass', (HazardCritic, HazardCriticCompetitive))
@param('actor_event_kwarg, actor_klass', [
    ('event', Actor),
    ('cond', CondActor)
])
@param('negate', (False, True))
def test_actor_learning(critic_klass, actor_event_kwarg, actor_klass, negate):
    torch.manual_seed(0)

    critic = critic_klass(
        dim = 16,
        depth = 2,
        dim_state = 4,
        num_actions = 7,
        pred_time_bins = 4,
        actor_event_kwarg = actor_event_kwarg
    )

    actor = actor_klass()

    optim = torch.optim.Adam(actor.parameters(), lr = 1e-2)

    state = torch.randn(2, 4)
    event = torch.randn(2, 4)

    before = critic.extract_policy(actor, state, event, negate = negate).mean().item()

    for _ in range(10):
        optim.zero_grad()
        values = critic.extract_policy(actor, state, event, negate = negate).mean()
        (-values).backward()
        optim.step()

    after = critic.extract_policy(actor, state, event, negate = negate).mean().item()

    assert after > before

def test_compute_first_dwell_time():
    goal = torch.tensor([[[1., 0.]]])
    next_states = torch.tensor([[[0., 0.], [1., 0.], [1., 0.], [0., 0.]]])

    reach, cutoff = compute_first_dwell_time(goal, next_states = next_states, eps = 0.5)
    assert reach.item() == 1
    assert cutoff.item() == 4

    # the same result when given the full states including the first

    states = torch.cat((torch.zeros(1, 1, 2), next_states), dim = 1)
    reach_states, cutoff_states = compute_first_dwell_time(goal, states = states, eps = 0.5)
    assert torch.equal(reach, reach_states)
    assert torch.equal(cutoff, cutoff_states)

    reach, cutoff = compute_first_dwell_time(goal, next_states = next_states, eps = 0.5, dwell_steps = 2)
    assert reach.item() == 1
    assert cutoff.item() == 3

    # batched multi-query

    goals = torch.tensor([1., 0.]).expand(2, 4, 2)

    next_states = torch.tensor([
        [[0., 0.], [1., 0.], [0., 0.], [1., 0.], [1., 0.], [1., 0.]],
        [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
    ])

    reach, cutoff = compute_first_dwell_time(goals, next_states = next_states, eps = 0.5, dwell_steps = 2)

    assert reach[0].tolist() == [3, 2, 1, 0]
    assert reach[1].tolist() == [-1, -1, -1, -1]
    assert cutoff.tolist() == [[5, 4, 3, 2], [5, 4, 3, 2]]

    # differing dwell steps per batch element

    reach, cutoff = compute_first_dwell_time(goals, next_states = next_states, eps = 0.5, dwell_steps = torch.tensor([1, 2]))

    assert reach[0].tolist() == [1, 0, 1, 0]
    assert reach[1].tolist() == [-1, -1, -1, -1]
    assert cutoff.tolist() == [[6, 5, 4, 3], [5, 4, 3, 2]]

    # variable lengths per batch element

    reach, cutoff = compute_first_dwell_time(
        goals,
        next_states = next_states,
        eps = 0.5,
        dwell_steps = 2,
        lens = torch.tensor([4, 6])
    )

    assert reach[0].tolist() == [-1, -1, -1, -1]
    assert reach[1].tolist() == [-1, -1, -1, -1]
    assert cutoff.tolist() == [[3, 2, 1, 0], [5, 4, 3, 2]]
