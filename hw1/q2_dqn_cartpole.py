# ============================================
# Section 2: DQN - CartPole-v1
# ============================================

import random
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace


# --------------------------
# Network architecture
# --------------------------

class QNetwork(nn.Module):
    """
    64 -> 32 -> 32 -> 24 -> 24 -> action_dim, ReLU activations.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
        )

        # Optional: He initialization similar to kernel_initializer='he_uniform'
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_network(lr: float, net_arch, device: torch.device):
    """
    Build a QNetwork + Adam optimizer.
    """
    model = QNetwork(net_arch.state_dim, net_arch.action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


# --------------------------
# Replay Buffer
# --------------------------

class ReplayBuffer:
    """
    Replay Buffer.
    Each experience: (state, action, next_state, reward, not_done)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List = []
        self.position = 0

    def store(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience
        self.position += 1

    def sample(self, batch_size: int):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([exp[0] for exp in mini_batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in mini_batch], dtype=np.int64)
        next_states = np.array([exp[2] for exp in mini_batch], dtype=np.float32)
        rewards = np.array([exp[3] for exp in mini_batch], dtype=np.float32)
        not_dones = np.array([exp[4] for exp in mini_batch], dtype=np.float32)
        return states, actions, next_states, rewards, not_dones

    def __len__(self):
        return len(self.memory)


# --------------------------
# DQN Agent
# --------------------------

class DQN_Agent:
    def __init__(self, hp, env_dim, device: torch.device):
        """
        DQN agent.

        hp: hyperparameters
        env_dim: environment dimensions
        """
        self.hp = hp
        self.net_arch = env_dim
        self.device = device

        self.epsilon = self.hp.max_epsilon
        self.memory = ReplayBuffer(capacity=self.hp.capacity)

        self.online_net, self.optimizer = build_network(
            lr=self.hp.lr, net_arch=self.net_arch, device=self.device
        )
        self.target_net, _ = build_network(
            lr=self.hp.lr, net_arch=self.net_arch, device=self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.loss_fn = nn.MSELoss()

    def store_experience(self, experience):
        self.memory.store(experience=experience)

    def choose_action(self, observation: np.ndarray):
        """
        Epsilon-greedy action selection.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore
            a = np.random.choice(self.net_arch.action_dim)
        else:
            # Exploit
            state_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.online_net(state_t)
            a = int(torch.argmax(q_vals, dim=1).item())
        self.epsilon_decay()
        return a

    def epsilon_decay(self):
        self.epsilon = max(self.hp.min_epsilon, self.epsilon * self.hp.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_step(self):
        """
        One gradient step of DQN:
        - sample batch
        - compute target
        - update online network
        Returns loss (float)
        """
        # Optional extra epsilon decay per training step
        self.epsilon_decay()

        if len(self.memory) < self.hp.batch_size:
            return 0.0

        states, actions, next_states, rewards, not_dones = self.memory.sample(self.hp.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        not_dones_t = torch.tensor(not_dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Q(s,a) for taken actions
        q_pred = self.online_net(states_t).gather(1, actions_t)

        # Q_target(s', a') from target network
        with torch.no_grad():
            q_next = self.target_net(next_states_t)
            max_q_next = torch.max(q_next, dim=1, keepdim=True)[0]
            q_target = rewards_t + self.hp.gamma * max_q_next * not_dones_t

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def test_agent(self, env, visualize=True):
        """
        Greedy evaluation using online network.
        """
        state, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if visualize:
                env.render()

            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.online_net(state_t)
            a = int(torch.argmax(q_vals, dim=1).item())

            next_state, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Total reward in test: {total_reward:.2f}")
        env.close()


# --------------------------
# Main Training Loop
# --------------------------

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    writer = SummaryWriter(log_dir="runs/dqn_cartpole/5_layers")

    # Environment dimensions
    env_dim_dict = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
    }
    env_dim = SimpleNamespace(**env_dim_dict)

    # Best hyperparameters from the reference (adapted) :contentReference[oaicite:3]{index=3}
    best_hyper_parameters = {
        'lr': 0.0001,
        'batch_size': 128,
        'capacity': 10_000,
        'gamma': 0.99,
        'max_epsilon': 0.9,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.999,
        'target_update_period': 100,
    }
    hp = SimpleNamespace(**best_hyper_parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    agent = DQN_Agent(hp=hp, env_dim=env_dim, device=device)

    max_episodes = 1000
    max_steps = 500
    max_score = 475.0  # solving threshold (100-episode moving avg)
    total_steps = 0

    episode_loss = []
    average_score = []

    for episode in range(max_episodes):
        state, info = env.reset()
        done = False
        episode_score = 0.0

        for step in range(max_steps):
            total_steps += 1

            action = agent.choose_action(observation=state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_score += reward

            not_done = 0.0 if done else 1.0
            agent.store_experience((state, action, next_state, reward, not_done))

            if done:
                average_score.append(episode_score)
                print(f"Episode: {episode + 1} | Score: {episode_score}")
                break

            state = next_state
            loss = agent.update_step()
            writer.add_scalar('Loss/train_step', loss, total_steps)
            episode_loss.append(loss)

            if (total_steps + 1) % hp.target_update_period == 0:
                agent.update_target_network()

        # TensorBoard episode logs
        writer.add_scalar('Reward/episode', episode_score, episode)
        if len(average_score) >= 100:
            mean_last_100 = float(np.mean(average_score[-100:]))
        else:
            mean_last_100 = float(np.mean(average_score))
        writer.add_scalar('Reward/mean_last_100', mean_last_100, episode)

        if (episode + 1) % 100 == 0:
            print(f"100-Episode Average Score: {mean_last_100:.2f}")

        if len(average_score) >= 100 and mean_last_100 >= max_score:
            print(
                "\nGreat!! "
                f"You solved the environment after {episode + 1} episodes.\n"
                f"Average reward over last 100 episodes: {mean_last_100:.2f}"
            )
            break

    writer.close()