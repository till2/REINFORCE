# REINFORCE implementation

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


# Hyperparams
EPISODES = 10_000
GAMMA = 0.99
ALPHA = 0.01
RENDER = True
WANDB = True


# Env setup
env = gym.make("CartPole-v1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

if WANDB:
    wandb.init(project="REINFORCE-CartPoleV1")

# Policy Gradient REINFORCE-Agent
class REINFORCE(nn.Module):

    def __init__(self, features, outputs):
        super(REINFORCE, self).__init__()
        layers = [
            nn.Linear(features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, outputs),
        ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.net(torch.Tensor(x))
        return x
    
    def act(self, x):
        out = self.forward(x)
        pd = torch.distributions.Categorical(logits=out)
        action = pd.sample()
        # print(action[0].detach().numpy(), pd.log_prob(action))
        return (action[0].detach().numpy(), pd.log_prob(action))


# Init Agent
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

pi = REINFORCE(features=obs_shape, outputs=action_shape)
print(pi)

optimizer = torch.optim.Adam(pi.net.parameters(), lr=ALPHA)


# Training loop
episode_rewards = []

for episode in range(EPISODES):

    # reset rewards & action probability distributions
    rewards = []
    action_log_probs = []

    # get one trajectory from current policy pi
    obs = env.reset()

    for t in range(200):

        if RENDER and episode%500 == 0:
            env.render()

        # get pd & action from current policy
        action, action_logprob = pi.act(obs[None, :])

        # take action to get new observation
        obs, reward, done, info = env.step(action)

        # store action log-probability distribution & reward
        action_log_probs.append(action_logprob)
        rewards.append(reward)

        if done:
            break

    if WANDB:
        wandb.log({"accumulated_reward": sum(rewards)})

    # calculating the gradient
    # followed the code from "Foundations of Deep RL - Addison Wesley" here
    T = len(rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0

    for t in reversed(range(T)):
        future_ret = rewards[t] + GAMMA * future_ret
        rets[t] = future_ret
    rets = torch.Tensor(rets)

    action_log_probs = torch.stack(action_log_probs)

    loss = - action_log_probs * rets
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env.close()
