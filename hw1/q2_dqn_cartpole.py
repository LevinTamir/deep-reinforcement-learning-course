import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from utils import QNetwork, ReplayBuffer, build_network, sample_action, save_plots

FIG_DIR = "DQN"
os.makedirs(FIG_DIR, exist_ok=True)

def train_agent(
    num_hidden_layers: int,
    hp: dict,
    state_dim: int,
    action_dim: int,
    run_name: str,
    log_dir: str,
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
    env.reset(seed=1)
    env.action_space.seed(1)

    online_net, optimizer = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net, _ = build_network(
        state_dim, action_dim, hp["lr"], device, num_hidden_layers
    )
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    buffer = ReplayBuffer(capacity=hp["capacity"])
    writer = SummaryWriter(log_dir=log_dir)

    epsilon = hp["max_epsilon"]
    loss_fn = nn.MSELoss()

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

            action = sample_action(online_net, state, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            not_done = 0.0 if done else 1.0
            buffer.store((state, action, next_state, reward, not_done))
            state = next_state

            # epsilon decay per step
            epsilon = max(hp["min_epsilon"], epsilon * hp["epsilon_decay"])

            # update only if enough samples
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
                    q_next = target_net(next_states_t)
                    max_q_next = q_next.max(dim=1, keepdim=True)[0]
                    q_targets = rewards_t + hp["gamma"] * max_q_next * not_dones_t

                loss = loss_fn(q_values, q_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar(f"{run_name}/loss_step", loss.item(), total_steps)
                episode_losses.append(loss.item())

                # update target network every C steps
                if total_steps % hp["target_update_period"] == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)

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

        if mean_last_100 >= max_score and best_solved_episode is None and len(episode_rewards) >= 100:
            best_solved_episode = episode + 1
            print(
                f"{run_name} solved after {best_solved_episode} episodes "
                f"(mean reward ≥ {max_score} over 100 episodes)"
            )

        # run ~50 more episodes after solve and then stop
        if best_solved_episode is not None and episode - best_solved_episode > 50:
            print(f"{run_name} stopping 50 episodes after solve")
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
    device = next(q_network.parameters()).device
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=123)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
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


def optimize_dqn(state_dim: int, action_dim: int,
                 max_episodes_sweep: int = 600):
    '''
    Run a small hyper-parameter sweep for 3 and 5 layer DQNs.
    
    '''

    # search ranges from the assignment
    lrs = [1e-4, 1e-5]
    gammas = [0.99, 0.999]
    batch_sizes = [64, 128]
    target_periods = [100, 200]

    base_hp = {
        "lr": 1e-4,             # will be overwritten
        "batch_size": 128,      # will be overwritten
        "capacity": 10_000,
        "gamma": 0.99,          # will be overwritten
        "max_epsilon": 1.0,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.999,
        "target_update_period": 100,  # will be overwritten
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
                        res = train_agent(
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

    # sort by best mean reward over 100 episodes
    results.sort(key=lambda r: r["best_mean_100"], reverse=True)

    # print top few configurations
    print("\nTop 5 configurations by best mean_100:")
    for r in results[:5]:
        hp = r["hp"]
        print(
            f"{r['run_name']} | depth={r['depth']} | "
            f"best_mean_100={r['best_mean_100']:.2f} | "
            f"lr={hp['lr']} gamma={hp['gamma']} bs={hp['batch_size']} tu={hp['target_update_period']}"
        )

    # 1) plot mean_100 curves for ALL runs (no legend)
    plt.figure(figsize=(10, 6))
    for r in results:
        x = np.arange(len(r["moving_avg_rewards"]))
        plt.plot(x, r["moving_avg_rewards"], alpha=0.3)
    plt.title("Q2 – Mean Reward (Last 100 Episodes) – All DQN Configurations")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.tight_layout()
    all_fig = os.path.join(FIG_DIR, "q2_hyperparam_sweep_all.png")
    plt.savefig(all_fig, dpi=200)
    plt.close()
    print(f"Saved sweep figure (all configs) to {all_fig}")

    # 2) plot mean_100 curves for the best few runs (with legend)
    top_k = min(5, len(results))
    plt.figure(figsize=(10, 6))
    for r in results[:top_k]:
        x = np.arange(len(r["moving_avg_rewards"]))
        label = (
            f"L{r['depth']}, lr={r['hp']['lr']}, "
            f"g={r['hp']['gamma']}, bs={r['hp']['batch_size']}, tu={r['hp']['target_update_period']}"
        )
        plt.plot(x, r["moving_avg_rewards"], label=label)

    plt.title("Q2 – Mean Reward (Last 100 Episodes) – Best DQN Configurations")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    best_fig = os.path.join(FIG_DIR, "q2_hyperparam_sweep_best.png")
    plt.savefig(best_fig, dpi=200)
    plt.close()
    print(f"Saved sweep comparison figure (best few) to {best_fig}")

    return results


if __name__ == "__main__":
    '''
    Entry point: trains final 3- and 5-layer DQNs and (optionally) runs a sweep.
    
    '''

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # hyperparameters chosen for the final solution (after running the sweep)
    best_hp = {
        "lr": 1e-4,
        "batch_size": 128,
        "capacity": 10_000,
        "gamma": 0.999,
        "max_epsilon": 0.9,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.999,
        "target_update_period": 100,
    }

    # train 3-layer network with full plots
    res_3 = train_agent(
        num_hidden_layers=3,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="q2_3_layers",
        log_dir="runs/q2_3_layers",
    )

    # train 5-layer network with full plots
    res_5 = train_agent(
        num_hidden_layers=5,
        hp=best_hp,
        state_dim=state_dim,
        action_dim=action_dim,
        run_name="q2_5_layers",
        log_dir="runs/q2_5_layers",
    )

    # run hyper-parameter sweep to tune the network
    # optimize_dqn(state_dim=state_dim, action_dim=action_dim, max_episodes_sweep=600)
