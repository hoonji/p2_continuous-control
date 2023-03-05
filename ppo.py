# MIT License

# Copyright (c) 2022 Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin,
                   # Anssi Kanervisto, Weixun Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified by Ji Hoon (Hoonji) Baek
# - modified to support unityagents multi agent Reacher environment
# - hyperparameter tuning
# - updated network architecture
# - removed non-essential code
# - simplified score recording
# - checkpoints model

import time
import pickle
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from ppo_agent import Agent

LEARNING_RATE = 3e-4
ADAM_EPS = 1e-5
TOTAL_TIMESTEPS = 4000000
GAMMA = .99
LAMBDA = .95
UPDATE_EPOCHS = 10
N_MINIBATCHES = 32
CLIP_COEF = .2
MAX_GRAD_NORM = 5
GAE_LAMBDA = .95
V_COEF = .5
ENT_COEF = .01
HIDDEN_LAYER_SIZE = 512
ANNEAL_LR = False
ROLLOUT_LEN = 2048


def run_ppo(env):
  """Trains a ppo agent in an environment.

  Saves model and learning curve checkpoints.
  """
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  env_info = env.reset(train_mode=True)[brain_name]
  num_agents = len(env_info.agents)
  n_observations = env_info.vector_observations.shape[1]
  n_actions = brain.vector_action_space_size
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  batch_size = ROLLOUT_LEN * num_agents

  agent = Agent(n_observations, n_actions, HIDDEN_LAYER_SIZE).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

  obs = torch.zeros((ROLLOUT_LEN, num_agents, n_observations)).to(device)
  actions = torch.zeros((ROLLOUT_LEN, num_agents, n_actions)).to(device)
  logprobs = torch.zeros(ROLLOUT_LEN, num_agents).to(device)
  rewards = torch.zeros(ROLLOUT_LEN, num_agents).to(device)
  advantages = torch.zeros(ROLLOUT_LEN, num_agents).to(device)
  dones = torch.zeros(ROLLOUT_LEN, num_agents).to(device)
  values = torch.zeros(ROLLOUT_LEN, num_agents).to(device)

  next_obs = torch.Tensor(env_info.vector_observations).to(device)
  next_done = torch.zeros(num_agents).to(device)
  num_updates = TOTAL_TIMESTEPS // batch_size
  minibatch_size = batch_size // N_MINIBATCHES
  current_returns = np.zeros(num_agents)
  scores = []
  time_checkpoint = time.time()

  for update in range(1, num_updates + 1):
    print(
        f"update {update}/{num_updates}. Last update in {time.time() - time_checkpoint}s"
    )
    time_checkpoint = time.time()
    # Anneal learning rate
    if ANNEAL_LR:
      frac = 1.0 - (update - 1.0) / num_updates
      optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

    for step in range(ROLLOUT_LEN):
      obs[step] = next_obs
      dones[step] = next_done

      # import pdb; pdb.set_trace()
      with torch.no_grad():
        cur_actions, cur_logprobs, _, cur_values = agent.get_action_and_value(next_obs)
      values[step] = cur_values.flatten()
      actions[step] = cur_actions
      logprobs[step] = cur_logprobs

      env_info = env.step(cur_actions.cpu().numpy())[brain_name]
      rewards[step] = torch.tensor(env_info.rewards).to(device)

      current_returns += env_info.rewards
      scores.extend(current_returns[env_info.local_done])
      current_returns[env_info.local_done] = 0

      next_obs = torch.Tensor(env_info.vector_observations).to(device)
      next_done = torch.Tensor([env_info.local_done]).to(device)

    with torch.no_grad():
      next_value = agent.get_value(next_obs).flatten()

    lastgaelam = 0
    for t in reversed(range(ROLLOUT_LEN)):
      if t == ROLLOUT_LEN - 1:
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
      else:
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
      delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
      advantages[
          t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
    returns = advantages + values

    b_obs = obs.reshape((-1, n_observations))
    b_actions = actions.reshape((-1, n_actions))
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(batch_size)
    for epoch in range(UPDATE_EPOCHS):
      np.random.shuffle(b_inds)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        mb_advantages = b_advantages[mb_inds]
        # Advantage normalization
        mb_advantages = (mb_advantages -
                         mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF,
                                                1 + CLIP_COEF)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds],
            -CLIP_COEF,
            CLIP_COEF,
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = v_loss_max.mean()

        entropy_loss = entropy.mean()
        loss = pg_loss + V_COEF * v_loss - ENT_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    torch.save(agent.state_dict(), f'checkpoints/model_checkpoint.pickle')
    with open(f'checkpoints/scores.pickle', 'wb') as f:
      pickle.dump(scores, f)

    print(f'last 100 returns: {np.array(scores[-100:]).mean()}')
