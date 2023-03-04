import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

class Agent(nn.Module):

  def __init__(self, n_observations, n_actions, hidden_size=256):
    super().__init__()
    self.critic = nn.Sequential(
        layer_init(
            nn.Linear(
                np.array(n_observations).prod(), hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, 1), std=1.0),
        nn.Tanh(),
    )
    self.actor_mean = nn.Sequential(
        layer_init(
            nn.Linear(
                np.array(n_observations).prod(), hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, np.prod(n_actions)),
                   std=0.01),
    )
    self.actor_logstd = nn.Parameter(
        torch.zeros(1, np.prod(n_actions)))

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, action=None):
    action_mean = self.actor_mean(x)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(
        1), self.critic(x)
