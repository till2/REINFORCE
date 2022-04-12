# REINFORCE implementation

import gym
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Hyperparams
EPISODES = 10_000
MOV_AVG = 1_000
GAMMA = 0.99
ALPHA = 0.01
RENDER = True
WANDB = True


# Env setup
env = gym.make("Blackjack-v0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if WANDB:
    wandb.init(project="REINFORCE-torch")

# New Agent
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

    """
    # calculate sum of discounted future reward efficiently
    episode_rewards.append(sum(rewards))
    ma = pandas.Series(episode_rewards).rolling(MOV_AVG).mean().tolist()[-1]
    if WANDB:
        if episode > MOV_AVG:
            wandb.log({"ma": np.float16(ma), "accumulated_reward": sum(rewards)})

    l = len(rewards)
    future_rewards = []
    for t in reversed(range(l)):
        if t == l-1:
            G = rewards[t]
        else:
            G = rewards[t] + GAMMA * future_rewards[l-t-2]

        future_rewards.append(G)
    future_rewards.reverse()
    
    # normalize future rewards
    future_rewards = (future_rewards - np.mean(future_rewards)) / (np.std(future_rewards) + 1e-20)

    # calculate gradient of policy parameters to backpropagate it
    pi_gradient = []

    for log_prob, future_reward in zip(action_log_probs, future_rewards):
        # append the negative gradient because we want to minimize it later,
        # that is the same as maximizing expected future rewards
        pi_gradient.append(torch.Tensor(- log_prob * future_reward))
    
    print("gradient:")
    print(pi_gradient)
    print()

    pi_gradient = torch.stack(pi_gradient)
    print("stacked:")
    print(pi_gradient)


    gradient_sum = sum(pi_gradient)
    print()
    print("sum:")
    print(gradient_sum)
    
    # update weights
    optimizer.zero_grad()
    loss = gradient_sum
    # loss.backward()
    optimizer.step()
    """
    # print(rewards)
    # print(action_log_probs)
    if WANDB:
        wandb.log({"accumulated_reward": sum(rewards)})
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



# Testing area
# print(env.action_space) # Discrete(2)
# print(env.observation_space.shape) # (4,)

# action: 0 or 1

# print(env.action_space.n) # 2
# print(env.observation_space.shape[0]) # 4


# pick actions via probability:
# action = np.random.choice(
#     action_shape,                               # actions are nums for gym envs
#     1,                                          # pick one action
#     p=agent.pi.predict(obs[None, :]).flatten()  # probability distribution for actions
# )[0]

env.close()