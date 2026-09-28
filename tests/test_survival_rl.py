import pytest
param = pytest.mark.parametrize

import torch
from torch.nn import Module

from x_mlps_pytorch import MLP
from survival_rl import HazardCritic

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

# test

@param('reach_event_index, horizon_cutoff, shape', [
    (None, None, (2, 4)),
    (torch.tensor([0, 1]), None, ()),
    (torch.tensor([1, 2]), torch.tensor([3, 2]), ()),
])
def test_critic(reach_event_index, horizon_cutoff, shape):
    critic = HazardCritic(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    assert critic(state, action, event, reach_event_index = reach_event_index, horizon_cutoff = horizon_cutoff).shape == shape

def test_negative_reach_index_is_censored():
    torch.manual_seed(0)

    critic = HazardCritic(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4)

    state = torch.randn(2, 4)
    action = torch.randn(2, 7)
    event = torch.randn(2, 4)

    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, 1]))
    censored = critic(state, action, event, reach_event_index = torch.tensor([4, 1]))

    assert torch.allclose(loss, censored)

    loss = critic(state, action, event, reach_event_index = torch.tensor([-1, 1]), horizon_cutoff = torch.tensor([2, 4]))
    censored = critic(state, action, event, reach_event_index = torch.tensor([2, 1]), horizon_cutoff = torch.tensor([2, 4]))

    assert torch.allclose(loss, censored)

def test_e2e():
    torch.manual_seed(0)

    critic = HazardCritic(dim = 16, depth = 2, dim_state = 4, num_actions = 7, pred_time_bins = 4, discount_factor = 0.9)
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

@param('actor_event_kwarg, actor_klass', [
    ('event', Actor),
    ('cond', CondActor)
])
@param('negate', (False, True))
def test_actor_learning(actor_event_kwarg, actor_klass, negate):
    torch.manual_seed(0)

    critic = HazardCritic(
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
