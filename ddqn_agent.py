import torch
import torch.nn as nn
import torch.optim as optim
import random

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, n):
        self.model = DDQN(n, n)
        self.target = DDQN(n, n)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.eps = 1.0

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, len(state) - 1)
        with torch.no_grad():
            return torch.argmax(self.model(torch.tensor(state).float())).item()

    def train(self, s, a, r, ns):
        s = torch.tensor(s).float()
        ns = torch.tensor(ns).float()
        q = self.model(s)[a]
        nq = self.target(ns).max()
        loss = (q - (r + 0.99 * nq)) ** 2

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.eps = max(0.1, self.eps * 0.995)
