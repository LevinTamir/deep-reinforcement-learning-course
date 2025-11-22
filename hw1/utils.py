import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt

class QNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_size: int, num_hidden_layers: int):
        super().__init__()

        layers = []
        in_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_network(state_dim: int, action_dim: int,
                  lr: float, device, num_hidden_layers: int):
    
    model = QNetwork(state_dim, action_dim,
                     hidden_size=128, num_hidden_layers=num_hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def sample_action(q_network: QNetwork,
                  state: np.ndarray,
                  epsilon: float,
                  action_dim: int,
                  device) -> int:

    if random.random() < epsilon:
        return random.randrange(action_dim)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals = q_network(state_t)
    return int(torch.argmax(q_vals, dim=1).item())


def save_plots(losses, rewards, moving_avg, run_name, PLT_DIR):

    x_loss = np.arange(len(losses))
    x_ep = np.arange(len(rewards))

    plt.figure(figsize=(8, 5))
    plt.plot(x_loss, losses)
    plt.title(f"{run_name} step loss")
    plt.xlabel("step num")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_DIR, f"{run_name}_step_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x_ep, rewards)
    plt.title(f"{run_name} reward per episode")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_DIR, f"{run_name}_reward.png"), dpi=200)
    plt.close()

    
    plt.figure(figsize=(8, 5))
    plt.plot(x_ep, moving_avg)
    plt.title(f"{run_name} mean reward last 100 episodes")
    plt.xlabel("episode")
    plt.ylabel("mean reward")
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_DIR, f"{run_name}_mean_reward_100.png"), dpi=200)
    plt.close()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return np.array(state), np.array(action), np.array(next_state), np.array(reward), np.array(done)

    def __len__(self):
        return len(self.buffer)
