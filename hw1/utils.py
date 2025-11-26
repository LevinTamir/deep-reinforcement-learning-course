# ============================================
# Utils file
# Mark Feldman (320827637) & Tamir Levin (315765347)
# ============================================

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


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list | int, num_hidden_layers: int = None):
        super().__init__()
        
        if isinstance(hidden_sizes, int):
            if num_hidden_layers is None:
                raise ValueError("If hidden_sizes is an int, num_hidden_layers must be provided")
            hidden_sizes = [hidden_sizes] * num_hidden_layers
        
        layers = []
        in_dim = state_dim
        
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.feature_layer = nn.Sequential(*layers)
        
        self.value_stream = nn.Linear(hidden_sizes[-1], 1)
        self.advantage_stream = nn.Linear(hidden_sizes[-1], action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals


def build_network(state_dim: int, action_dim: int,
                  lr: float, device, num_hidden_layers: int):
    
    model = QNetwork(state_dim, action_dim,
                     hidden_size=128, num_hidden_layers=num_hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def build_dueling_network(state_dim: int, action_dim: int,
                          lr: float, device, num_hidden_layers: int):
    
    model = DuelingQNetwork(state_dim, action_dim,
                            hidden_sizes=128, num_hidden_layers=num_hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def sample_action(q_network: nn.Module,
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


def optimize_dqn(train_agent_fn,
                 state_dim: int,
                 action_dim: int,
                 max_episodes_sweep: int = 600,
                 fig_dir: str = "DQN"):

    lrs = [1e-4, 1e-5]
    gammas = [0.99, 0.999]
    batch_sizes = [64, 128]
    target_periods = [100, 200]

    base_hp = {
        "lr": 1e-4,
        "batch_size": 128,
        "capacity": 50_000,
        "gamma": 0.99,
        "max_epsilon": 1.0,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.9999,
        "target_update_period": 100,
    }

    results = []

    for depth in (3, 5):
        for lr in lrs:
            for gamma in gammas:
                for bs in batch_sizes:
                    for tu in target_periods:
                        hp = base_hp.copy()
                        hp["lr"] = lr
                        hp["gamma"] = gamma
                        hp["batch_size"] = bs
                        hp["target_update_period"] = tu

                        run_name = f"sweep_L{depth}_lr{lr}_g{gamma}_bs{bs}_tu{tu}"
                        log_dir = f"runs/q2_sweep/{run_name}"

                        print(f"\n=== starting sweep run: {run_name} ===")
                        res = train_agent_fn(
                            num_hidden_layers=depth,
                            hp=hp,
                            state_dim=state_dim,
                            action_dim=action_dim,
                            run_name=run_name,
                            log_dir=log_dir,
                            max_episodes=max_episodes_sweep,
                        )
                        res["run_name"] = run_name
                        res["depth"] = depth
                        res["hp"] = hp
                        results.append(res)

    results.sort(key=lambda r: r["best_mean_100"], reverse=True)

    print("\nTop 5 configurations by best mean_100:")
    for r in results[:5]:
        hp = r["hp"]
        print(
            f"{r['run_name']} | depth={r['depth']} | "
            f"best_mean_100={r['best_mean_100']:.2f} | "
            f"lr={hp['lr']} gamma={hp['gamma']} bs={hp['batch_size']} tu={hp['target_update_period']}"
        )

    plt.figure(figsize=(10, 6))
    for r in results:
        x = np.arange(len(r["moving_avg_rewards"]))
        plt.plot(x, r["moving_avg_rewards"], alpha=0.3)
    plt.title("Mean Reward (Last 100 Episodes) All DQN Configurations")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.tight_layout()
    all_fig = os.path.join(fig_dir, "q2_hyperparam_sweep_all.png")
    plt.savefig(all_fig, dpi=200)
    plt.close()
    print(f"Saved sweep figure (all configs) to {all_fig}")

    top_k = min(5, len(results))
    plt.figure(figsize=(10, 6))
    for r in results[:top_k]:
        x = np.arange(len(r["moving_avg_rewards"]))
        label = (
            f"L{r['depth']}, lr={r['hp']['lr']}, "
            f"g={r['hp']['gamma']}, bs={r['hp']['batch_size']}, tu={r['hp']['target_update_period']}"
        )
        plt.plot(x, r["moving_avg_rewards"], label=label)

    plt.title("Q2 - Mean Reward (Last 100 Episodes) - Best DQN Configurations")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    best_fig = os.path.join(fig_dir, "q2_hyperparam_sweep_best.png")
    plt.savefig(best_fig, dpi=200)
    plt.close()
    print(f"Saved sweep comparison figure (best few) to {best_fig}")

    return results
