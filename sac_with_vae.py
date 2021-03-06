import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import datetime

from collections import deque

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, linear_output, act_size, hidden_size):
        super().__init__()
        self.net1 = MLP(linear_output+act_size,1, hidden_size)
        self.net2 = MLP(linear_output+act_size,1, hidden_size)

    def forward(self, state, action):
        embedding = state
        state_action = torch.cat([embedding, action], 1)
        return self.net1(state_action), self.net2(state_action)

class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self, linear_output, act_size, hidden_size):
        super().__init__()
        self.act_size = act_size
        self.net = MLP(linear_output, act_size*2, hidden_size)

    def forward(self, state):
        x = self.net(state)
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
    def __init__(self, parameters):

        params = {
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.0001,
            "replay_buffer_size": 1000000,
            "hidden_size": 100,
            "batch_size": 64,
            "n_episodes": 1000,
            "n_random_episodes": 10,
            "discount": 0.9,
            "horizon": 50,
            "im_rows": 40,
            "im_cols": 40,
            "linear_output": 64,
            "target_entropy": -2
        }
        
        for arg in parameters:
            params[arg] = parameters[arg]


        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.lr = params["lr"]
        self.replay_buffer_size = params["replay_buffer_size"]
        self.hidden_size = params["hidden_size"]
        self.batch_size = params["batch_size"]
        self.n_episodes = params["n_episodes"]
        self.n_random_episodes = params["n_random_episodes"]
        self.discount = params["discount"]
        self.horizon = params["horizon"]
        self.im_rows = params["im_rows"]
        self.im_cols = params["im_cols"]
        self.linear_output = params["linear_output"]
        self.target_entropy = params["target_entropy"]
        self.act_size = 2

        self.critic = Critic(self.linear_output, self.act_size, self.hidden_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = Critic(self.linear_output, self.act_size, self.hidden_size).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor(self.linear_output, self.act_size, self.hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        

    def update_parameters(self, batch):

        #k = min(self.batch_size, len(self.replay_buffer))
        #batch = random.sample(self.replay_buffer, k=k)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(device) for t in zip(*batch)]

        alpha = 0.1#self.log_alpha.exp().item()

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

    def save_model(self, path="checkpoint"):
        
        save_path = "sac_model" + path + ".pth"

        data = {}
        data["actor_model"] = self.actor.state_dict()
        data["critic_model"] = self.critic.state_dict()
        data["critic_target"] = self.critic_target.state_dict()
        data["actor_optimizer"] = self.actor_optimizer.state_dict()
        data["critic_optimizer"] = self.critic_optimizer.state_dict()
        data["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        data["log_alpha"] = self.log_alpha

        torch.save(data, save_path)

    def load_model(self, path="sac_model_checkpoint.pth"):

        data = torch.load(path)
        self.actor.load_state_dict(data["actor_model"])
        self.critic.load_state_dict(data["critic_model"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])
        self.log_alpha = data["log_alpha"]

