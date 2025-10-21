
import torch, torch.nn as nn

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.V = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, q_vals, state):
        bs = q_vals.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, -1)
        w2 = torch.abs(self.hyper_w2(state)).view(bs, -1, 1)
        hidden = torch.bmm(q_vals.unsqueeze(1), w1).squeeze(1)
        y = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + self.V(state)
        return y
