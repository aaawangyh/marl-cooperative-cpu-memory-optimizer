
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, n_agents, cfg):
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.01)
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim*n_agents, act_dim*n_agents)
        self.targ_actor = Actor(obs_dim, act_dim)
        self.targ_critic = Critic(obs_dim*n_agents, act_dim*n_agents)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=cfg.get("lr_actor", 1e-4))
        self.opt_c = optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.noise_std = cfg.get("noise_std", 0.2)

    def act(self, obs):
        with torch.no_grad():
            a = self.actor(torch.tensor(obs, dtype=torch.float32))
            a += self.noise_std * torch.randn_like(a)
            return a.clamp(-1,1).numpy()
