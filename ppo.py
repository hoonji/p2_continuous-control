import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random
from ppo_agent import Agent
import time
import pickle
from collections import deque

LEARNING_RATE = 3e-4
ADAM_EPS = 1e-5
BATCH_SIZE = 2048
TOTAL_TIMESTEPS = 2000000
GAMMA = .99
LAMBDA = .95
UPDATE_EPOCHS = 10
N_MINIBATCHES = 32
CLIP_COEF = .2
MAX_GRAD_NORM = .5
GAE_LAMBDA = .95
V_COEF = .5
ENT_COEF = .01

def run_ppo(env):
  """Trains a ppo agent in an environment.

  Saves model and learning curve checkpoints.
  """
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  env_info = env.reset(train_mode=True)[brain_name]
  n_observations = env_info.vector_observations.shape[1]
  n_actions = brain.vector_action_space_size
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  agent = Agent(n_observations, n_actions).to(device)
  #print(agent)
  optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

  obs = torch.zeros((BATCH_SIZE, n_observations)).to(device)
  actions = torch.zeros((BATCH_SIZE, n_actions)).to(device)
  logprobs = torch.zeros(BATCH_SIZE).to(device)
  rewards = torch.zeros(BATCH_SIZE).to(device)
  advantages = torch.zeros(BATCH_SIZE).to(device)
  dones = torch.zeros(BATCH_SIZE).to(device)
  values = torch.zeros(BATCH_SIZE).to(device)

  next_obs = torch.Tensor(env_info.vector_observations[0]).to(device)
  next_done = torch.zeros(1).to(device)
  num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
  minibatch_size = BATCH_SIZE // N_MINIBATCHES
  total_rewards = [0]
  episode_steps = [0]
  time_checkpoint = time.time()

  for update in range(1, num_updates + 1):
    print(
        f"update {update}/{num_updates}. Last update in {time.time() - time_checkpoint}s"
    )
    time_checkpoint = time.time()

    for step in range(BATCH_SIZE):
      obs[step] = next_obs
      dones[step] = next_done

      with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(
            obs[step].expand(1, -1))
      values[step] = value.flatten()
      actions[step] = action.flatten()
      logprobs[step] = logprob

      action = np.expand_dims(np.clip(actions[step].cpu().numpy(), -1, 1), 1)
      env_info = env.step(action)[brain_name]
      reward = env_info.rewards[0]
      rewards[step] = torch.tensor(reward).to(device)

      total_rewards[-1] += reward
      episode_steps[-1] += 1
      if env_info.local_done[0]:
        total_rewards.append(0)
        episode_steps.append(0)
      next_obs = torch.Tensor(env_info.vector_observations[0]).to(device)
      next_done = torch.Tensor([env_info.local_done[0]]).to(device)

    with torch.no_grad():
      next_value = agent.get_value(next_obs).flatten()

    lastgaelam = 0
    for t in reversed(range(BATCH_SIZE)):
      if t == BATCH_SIZE - 1:
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
      else:
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
      delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
      advantages[
          t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
    returns = advantages + values

    b_inds = np.arange(BATCH_SIZE)

    for epoch in range(UPDATE_EPOCHS):
      np.random.shuffle(b_inds)
      for start in range(0, BATCH_SIZE, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            obs[mb_inds], actions[mb_inds])
        logratio = newlogprob - logprobs[mb_inds]
        ratio = logratio.exp()

        mb_advantages = advantages[mb_inds]
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
        v_loss_unclipped = (newvalue - returns[mb_inds])**2
        v_clipped = values[mb_inds] + torch.clamp(
            newvalue - values[mb_inds],
            -CLIP_COEF,
            CLIP_COEF,
        )
        v_loss_clipped = (v_clipped - returns[mb_inds])**2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        entropy_loss = entropy.mean()
        loss = pg_loss + V_COEF * v_loss - ENT_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    torch.save(agent.state_dict(), f'checkpoints/model_step_{update}.pickle')
    with open(f'checkpoints/eplen_and_returns_{update}.pickle', 'wb') as f:
      pickle.dump([(steps, r)
                   for steps, r in zip(episode_steps, total_rewards)], f)

    print(f'last 100 returns: {np.array(total_rewards[-100:]).mean()}')
