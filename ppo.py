# Implementation based on Costa Huang's PPO implementation: https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_continuous_action.py

import time
import pickle
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_agent import Agent

LEARNING_RATE = 3e-4
ADAM_EPS = 1e-5
TOTAL_TIMESTEPS = 2000000
GAMMA = .99
LAMBDA = .95
UPDATE_EPOCHS = 10
N_MINIBATCHES = 32
CLIP_COEF = .2
MAX_GRAD_NORM = 5
GAE_LAMBDA = .95
V_COEF = .5
HIDDEN_LAYER_SIZE = 512
ROLLOUT_LEN = 2048

Rollout = namedtuple(
    'Rollout',
    ['observations', 'actions', 'logprobs', 'rewards', 'dones', 'values'])
Batch = namedtuple(
    'Batch', ['observations', 'actions', 'advantages', 'returns', 'logprobs'])


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

  rollout = Rollout(observations=torch.zeros(
      (ROLLOUT_LEN, num_agents, n_observations)).to(device),
                    actions=torch.zeros(
                        (ROLLOUT_LEN, num_agents, n_actions)).to(device),
                    logprobs=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    rewards=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    dones=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    values=torch.zeros(ROLLOUT_LEN, num_agents).to(device))

  next_observations = torch.Tensor(env_info.vector_observations).to(device)
  next_dones = torch.zeros(num_agents).to(device)
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

    for step in range(ROLLOUT_LEN):
      observations = next_observations

      with torch.no_grad():
        actions, logprobs = agent.get_actions_and_logprobs(
            observations)
        values = agent.predict_values(observations)
      env_info = env.step(actions.cpu().numpy())[brain_name]
      dones = env_info.local_done

      rollout.observations[step] = observations
      rollout.actions[step] = actions
      rollout.rewards[step] = torch.tensor(env_info.rewards).to(device)
      rollout.dones[step] = next_dones  # record previous dones for this step
      rollout.logprobs[step] = logprobs
      rollout.values[step] = values.flatten()

      # Record agent returns
      current_returns += env_info.rewards
      scores.extend(current_returns[dones])
      current_returns[dones] = 0

      next_observations = torch.Tensor(env_info.vector_observations).to(device)
      next_dones = torch.Tensor([dones]).to(device)

    advantages = torch.zeros(ROLLOUT_LEN, num_agents).to(device)
    z = 0
    for t in reversed(range(ROLLOUT_LEN)):
      if t == ROLLOUT_LEN - 1:
        next_nonterminal = 1.0 - next_dones
        with torch.no_grad():
          next_values = agent.predict_values(next_observations).flatten()
      else:
        next_nonterminal = 1.0 - rollout.dones[t + 1]
        next_values = rollout.values[t + 1]
      td_errors = rollout.rewards[
          t] + next_nonterminal * GAMMA * next_values - rollout.values[t]
      z = td_errors + next_nonterminal * GAMMA * GAE_LAMBDA * z
      advantages[t] = z

    # Reshape rollout variables for training
    batch = Batch(observations=rollout.observations.reshape(
        (-1, n_observations)),
                  actions=rollout.actions.reshape((-1, n_actions)),
                  advantages=advantages.reshape(-1),
                  returns=(advantages + rollout.values).reshape(-1),
                  logprobs=rollout.logprobs.reshape(-1))

    batch_indices = np.arange(batch_size)
    for epoch in range(UPDATE_EPOCHS):
      np.random.shuffle(batch_indices)
      for start in range(0, batch_size, minibatch_size):
        mbatch_indices = batch_indices[start:start + minibatch_size]
        minibatch = Batch(observations=batch.observations[mbatch_indices],
                          actions=batch.actions[mbatch_indices],
                          advantages=batch.advantages[mbatch_indices],
                          returns=batch.advantages[mbatch_indices],
                          logprobs=batch.advantages[mbatch_indices])

        _, logprobs = agent.get_actions_and_logprobs(minibatch.observations,
                                                    minibatch.actions)
        values = agent.predict_values(minibatch.observations)
        logratios = logprobs - minibatch.logprobs
        ratios = logratios.exp()

        # Surrogate objective
        pg_loss1 = -minibatch.advantages * ratios
        pg_loss2 = -minibatch.advantages * torch.clamp(ratios, 1 - CLIP_COEF,
                                                       1 + CLIP_COEF)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        value_loss = ((values - minibatch.returns)**2).mean()
        loss = pg_loss + V_COEF * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    torch.save(agent.state_dict(), f'model_checkpoint.pickle')
    with open(f'scores.pickle', 'wb') as f:
      pickle.dump(scores, f)

    print(f'last 100 returns: {np.array(scores[-100:]).mean()}')
