import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):

  def __init__(self, n_observations, n_actions, hidden_size=256):
    super().__init__()
    self.critic = nn.Sequential(
        nn.Linear(n_observations, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.Tanh(),
    )
    self.actor_means = nn.Sequential(nn.Linear(n_observations, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, n_actions))
    self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))

  def predict_values(self, x):
    return self.critic(x)

  def get_actions_and_logprobs(self, x, action=None):
    action_means = self.actor_means(x)
    action_stddevs = torch.exp(self.actor_logstd.expand_as(action_means))
    probs = Normal(action_means, action_stddevs)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action).sum(1)
