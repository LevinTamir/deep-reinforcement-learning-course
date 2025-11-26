# ============================================
# Section 2 – DQN (CartPole-v1)
# Mark Feldman (320827637) & Tamir Levin (315765347)
# ============================================

import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import QNetwork, ReplayBuffer, build_network, sample_action, save_plots, optimize_dqn

FIG_DIR = "DQN"
os.makedirs(FIG_DIR, exist_ok=True)


def train_agent(
    num_hidden_layers: int,
    hp: dict,
    state_dim: int,
    action_dim: int,
    run_name: str,
    log_dir: str = None,
    max_episodes: int = 600,
    max_steps: int = 500,
    max_score: float = 475.0,
    random_seed: int = 42
):
    '''
    Train a DQN agent on CartPole.
    Returns training stats and the final network.
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{run_name} using device: {device}")

    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    env = gym.make("CartPole-v1")
    env.reset(seed=1)
    env.action_space.seed(1)

    # Create online network and target network
    online_net, optimizer = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net, _ = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    buffer = ReplayBuffer(capacity=hp["capacity"])
    writer = SummaryWriter(log_dir=log_dir if log_dir else f"runs/{run_name}")

    epsilon = hp["max_epsilon"]
    loss_fn = nn.MSELoss()

    # Setup training variables
    total_steps = 0
    episode_losses = []
    episode_rewards = []
    moving_avg_rewards = []
    best_solved_episode = None

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        for _ in range(max_steps):
            total_steps += 1

            # Epsilon-greedy action selection
            action = sample_action(online_net, state, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Store transition in replay buffer
            not_done = 0.0 if terminated else 1.0
            buffer.store((state, action, next_state, reward, not_done))
            state = next_state

            # Decay epsilon
            epsilon = max(hp["min_epsilon"], epsilon * hp["epsilon_decay"])

            if len(buffer) >= hp["batch_size"]:
                # Sample a batch from replay buffer
                states, actions, next_states, rewards, not_dones = buffer.sample(
                    hp["batch_size"]
                )

                # Convert to tensors
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                not_dones_t = torch.tensor(not_dones, dtype=torch.float32, device=device).unsqueeze(-1)

                # Get Q-values for actions taken
                q_values = online_net(states_t).gather(1, actions_t)

                # Compute target Q-values using target network
                with torch.no_grad():
                    q_next = target_net(next_states_t)
                    max_q_next = q_next.max(dim=1, keepdim=True)[0]
                    q_targets = rewards_t + hp["gamma"] * max_q_next * not_dones_t

                # Compute loss and update online network
                loss = loss_fn(q_values, q_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar(f"{run_name}/loss_step", loss.item(), total_steps)
                episode_losses.append(loss.item())

                # Periodically sync target network with online network
                if total_steps % hp["target_update_period"] == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)

        # Compute moving average over last 100 episodes
        if len(episode_rewards) >= 100:
            mean_last_100 = float(np.mean(episode_rewards[-100:]))
        else:
            mean_last_100 = float(np.mean(episode_rewards))

        moving_avg_rewards.append(mean_last_100)

        writer.add_scalar(f"{run_name}/reward_episode", ep_reward, episode)
        writer.add_scalar(f"{run_name}/reward_mean_100", mean_last_100, episode)
        writer.add_scalar(f"{run_name}/epsilon", epsilon, episode)

        print(
            f"{run_name} ep {episode+1}  "
            f"reward={ep_reward:.1f}  mean_100={mean_last_100:.1f}  "
            f"eps={epsilon:.3f}"
        )

        # Check if environment is solved
        if mean_last_100 >= max_score and len(episode_rewards) >= 100:
            best_solved_episode = episode + 1
            print(
                f"{run_name} solved after {best_solved_episode} episodes "
                f"(mean reward ≥ {max_score} over 100 episodes)"
            )
            break

    env.close()
    writer.close()

    save_plots(episode_losses, episode_rewards, moving_avg_rewards, run_name, FIG_DIR)

    best_mean_100 = max(moving_avg_rewards) if moving_avg_rewards else 0.0

    return {
        "online_net": online_net,
        "episode_rewards": episode_rewards,
        "moving_avg_rewards": moving_avg_rewards,
        "episode_losses": episode_losses,
        "solved_at": best_solved_episode,
        "best_mean_100": best_mean_100,
    }


def test_agent(q_network: QNetwork, episodes: int = 5, render: bool = False):
    '''
    Evaluate a trained Q-network without exploration.
    '''
    device = next(q_network.parameters()).device
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=123)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Greedy action selection (no exploration)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_network(state_t)
            action = int(torch.argmax(q_vals, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"[test] episode {ep+1} reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":

    # Make environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # Best hyperparameters found from sweep
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

    # Train with 3 hidden layers
    res_3 = train_agent(
        num_hidden_layers=3,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="dqn_3_layers",
    )

    # Train with 5 hidden layers
    res_5 = train_agent(
        num_hidden_layers=5,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="dqn_5_layers",
    )

    # Hyperparameter sweep (optional)
    # optimize_dqn(train_agent, state_dim, action_dim, max_episodes_sweep=600, fig_dir=FIG_DIR)