from utils import save_plots
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

PLT_DIR = "DDQN"
os.makedirs(PLT_DIR, exist_ok=True)

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, experience):

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience
        self.position += 1

    def sample(self, batch_size: int):

        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int64)
        next_states = np.array([e[2] for e in batch], dtype=np.float32)
        rewards = np.array([e[3] for e in batch], dtype=np.float32)
        not_dones = np.array([e[4] for e in batch], dtype=np.float32)
        return states, actions, next_states, rewards, not_dones

    def __len__(self):
        return len(self.memory)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.features = nn.Sequential(*layers)
        
        self.value_stream = nn.Linear(hidden_sizes[-1], 1)
        
        self.advantage_stream = nn.Linear(hidden_sizes[-1], action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals


def build_network(state_dim: int, action_dim: int,
                  lr: float, device, num_hidden_layers: int = None):
    
    if num_hidden_layers == 5:
        hidden_sizes = [64, 64, 32, 32, 16]
    elif num_hidden_layers == 3:
        hidden_sizes = [64, 64, 32]
    else:
        hidden_sizes = [128] * (num_hidden_layers if num_hidden_layers else 3)

    model = DuelingQNetwork(state_dim, action_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def sample_action(q_network: DuelingQNetwork,
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



def train_agent(
    num_hidden_layers: int,
    hp: dict,
    state_dim: int,
    action_dim: int,
    run_name: str,
    max_episodes: int = 600,
    max_steps: int = 500,
    max_score: float = 475.0,
    random_seed: int = 42
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{run_name} using device: {device}")

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    online_net, optimizer = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net, _ = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    buffer = ReplayBuffer(capacity=hp["capacity"])

    epsilon = hp["max_epsilon"]
    loss_fn = nn.MSELoss()

    total_steps = 0
    episode_losses = []
    episode_rewards = []
    moving_avg_rewards = []
    best_solved_episode = None
    max_mean_reward = -float('inf')

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        for _ in range(max_steps):
            total_steps += 1

            action = sample_action(online_net, state, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            not_done = 0.0 if terminated else 1.0
            buffer.store((state, action, next_state, reward, not_done))

            if done:
                break

            state = next_state

            epsilon = max(hp["min_epsilon"], epsilon * hp["epsilon_decay"])

            if len(buffer) >= hp["batch_size"]:
                states, actions, next_states, rewards, not_dones = buffer.sample(
                    hp["batch_size"]
                )

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                not_dones_t = torch.tensor(not_dones, dtype=torch.float32, device=device).unsqueeze(-1)

                q_values = online_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    q_next_online = online_net(next_states_t)
                    next_actions = q_next_online.argmax(dim=1, keepdim=True)
                    
                    q_next_target = target_net(next_states_t)
                    max_q_next = q_next_target.gather(1, next_actions)
                    
                    q_targets = rewards_t + hp["gamma"] * max_q_next * not_dones_t

                loss = loss_fn(q_values, q_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                episode_losses.append(loss.item())

                if total_steps % hp["target_update_period"] == 0:
                    target_net.load_state_dict(online_net.state_dict())

        episode_rewards.append(ep_reward)

        if len(episode_rewards) >= 100:
            mean_last_100 = float(np.mean(episode_rewards[-100:]))
        else:
            mean_last_100 = float(np.mean(episode_rewards))

        moving_avg_rewards.append(mean_last_100)

        print(
            f"{run_name} ep {episode+1}  "
            f"reward:{ep_reward:.1f}  mean_100:{mean_last_100:.1f}  "
            f"eps:{epsilon:.3f}"
        )

        if mean_last_100 > max_mean_reward:
            max_mean_reward = mean_last_100
            save_path = os.path.join(PLT_DIR, f"{run_name}_best.pth")
            torch.save(online_net.state_dict(), save_path)
            print(f"  New best mean_100: {max_mean_reward:.1f} -> saved model to {save_path}")

        if mean_last_100 >= max_score and best_solved_episode is None and len(episode_rewards) >= 100:
            best_solved_episode = episode + 1
            print(
                f"{run_name} solved after {best_solved_episode} episodes "
                f"mean reward ≥ {max_score} over 100 episodes"
            )

        if best_solved_episode is not None and episode - best_solved_episode > 50:
            print(f"{run_name} stopping 50 episodes after solve")
            break

    env.close()

    save_plots(episode_losses, episode_rewards, moving_avg_rewards, run_name, PLT_DIR)

    best_mean_100 = max(moving_avg_rewards) if moving_avg_rewards else 0.0

    return {
        "online_net": online_net,
        "episode_rewards": episode_rewards,
        "moving_avg_rewards": moving_avg_rewards,
        "episode_losses": episode_losses,
        "solved_at": best_solved_episode,
        "best_mean_100": best_mean_100,
    }


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    best_hp = {
        "lr": 1e-4,
        "batch_size": 128,
        "capacity": 10_000,
        "gamma": 0.999,
        "max_epsilon": 1,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.999,
        "target_update_period": 100,
    }

    res_3 = train_agent(
        num_hidden_layers=3,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="ddqn_3_layers",
    )

    res_5 = train_agent(
        num_hidden_layers=5,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="ddqn_5_layers",
    )

    r3 = np.array(res_3["episode_rewards"])
    r5 = np.array(res_5["episode_rewards"])

    min_len = min(len(r3), len(r5))
    r3 = r3[:min_len]
    r5 = r5[:min_len]

    diff = r5 - r3
    
    plt.figure(figsize=(10, 6))
    plt.plot(diff, label="5-Layer - 3-Layer Reward Difference", color='purple')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Reward Difference (5-Layer minus 3-Layer)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Difference")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(PLT_DIR, "reward_difference.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\nComparison Plot Saved to: {save_path}")
    print(f"Total Absolute Difference: {np.sum(np.abs(diff)):.1f}")
    if np.sum(np.abs(diff)) == 0:
        print("WARNING: Runs are IDENTICAL.")
    else:
        print("SUCCESS: Runs are DIFFERENT.")