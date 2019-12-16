import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random

from collections import deque

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1),
            Flatten()
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.net[-1].weight.data.uniform_(-init_w, init_w)
        self.net[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self, linear_output, hidden_size, act_size):
        super().__init__()
        self.conv = Conv(linear_output)
        self.net1 = MLP(linear_output+act_size,1, hidden_size)
        self.net2 = MLP(linear_output+act_size,1, hidden_size)

    def forward(self, state, action):
        embedding = self.conv.forward(state)
        state_action = torch.cat([embedding, action], 1)
        return self.net1(state_action), self.net2(state_action)

class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self, linear_output, act_size, hidden_size):
        super().__init__()
        self.act_size = act_size
        self.conv = Conv(linear_output)
        self.net = MLP(linear_output, act_size*2, hidden_size)

    def forward(self, state):
        x = self.net(self.conv(state))
        mean, log_std = x[:, :self.act_size], x[:, self.act_size:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()

        # Enforcing action bounds
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def select_action(self, state):
        
        state = torch.FloatTensor(state).to(device)
        action, _ = self.sample(state)
        
        return action[0].detach().cpu().numpy()


class SAC:
    def __init__(self, gamma=0.99, tau=0.005, lr=0.001, replay_buffer_size=1000000, 
    hidden_size=256, batch_size=64, n_episodes=1000, n_random_episodes=10,
    discount=0.90, horizon=50, throttle_min=0, throttle_max=1, reward="speed",
    im_rows=40, im_cols=40, linear_output=64):

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.replay_buffer_size = replay_buffer_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_random_episodes = n_random_episodes
        self.discount = discount
        self.horizon = horizon
        self.throttle_max = throttle_max
        self.throttle_min = throttle_min
        self.im_rows = im_rows
        self.im_cols = im_cols
        self.linear_ouput = linear_output

        self.act_size = 2

        self.critic = critic = Critic(linear_output, hidden_size, self.act_size).to(device)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

        self.critic_target = Critic(linear_output, hidden_size, self.act_size).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor(linear_output, self.act_size, hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -self.act_size

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        

    def update_parameters(self):

        batch = random.sample(self.replay_buffer, k=self.batch_size)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(device) for t in zip(*batch)]

        alpha = self.log_alpha.exp().item()

        # Update critic

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_next = q_next - alpha * next_action_log_prob
            q_target = reward + not_done * self.gamma * value_next

        q1, q2 = self.critic(state, action)
        q1_loss = 0.5*F.mse_loss(q1, q_target)
        q2_loss = 0.5*F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

        # Update actor

        action_new, action_new_log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha*action_new_log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha

        alpha_loss = -(self.log_alpha * (action_new_log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def select_action(self, state):
        return self.actor.select_action(state)

    def push_buffer(self, e):

        self.replay_buffer.append(e)

    def process_image(self, rgb):

        im = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        obs = cv2.resize(im, (self.im_rows, self.im_cols))
        
        return obs

        