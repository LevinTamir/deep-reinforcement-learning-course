# ============================================
# Section 1 – Tabular Q-Learning (FrozenLake-v1)
# ============================================

import os
import random
from collections import namedtuple, OrderedDict
from itertools import product

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# Directory for saving all figures
FIG_DIR = "figs_q1"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Generate combinations of hyperparameters (used for the sweep)
# ---------------------------------------------------------------------
class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run", params.keys())
        return [Run(*v) for v in product(*params.values())]


# ---------------------------------------------------------------------
# GLIE epsilon schedule (linear or exponential)
# ---------------------------------------------------------------------
def get_glie(num_episodes, max_epsilon, decay_factor, linear=True):
    eps_min = 0.001
    if linear:
        return [
            max(eps_min, max_epsilon - ((1.0 - decay_factor) * i))
            for i in range(num_episodes)
        ]
    else:
        return [
            max(eps_min, max_epsilon * (decay_factor**i))
            for i in range(num_episodes)
        ]


# ---------------------------------------------------------------------
# Save plots: Q-table + reward curves
# ---------------------------------------------------------------------
def plot_graphs(q, title, heat=False, out_prefix=None, **kwargs):
    figsize = (8, 6)

    # Q-table heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(q, cmap="Greens", aspect="auto")
    plt.colorbar(im, label="Q-value")
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.tight_layout()
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_qtable.png", dpi=200)
    plt.close()

    if heat:
        return

    rewards = kwargs.get("rewards")
    avg_reward = kwargs.get("avg_reward")
    steps = kwargs.get("steps")

    # Reward per episode
    if rewards is not None:
        plt.figure(figsize=figsize)
        plt.plot(rewards)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        if out_prefix is not None:
            plt.savefig(f"{out_prefix}_rewards.png", dpi=200)
        plt.close()

    # Mean reward and mean steps (last 100 episodes)
    if avg_reward is not None and steps is not None:
        x_vals = np.arange(len(steps)) * 100

        plt.figure(figsize=figsize)
        plt.plot(x_vals, avg_reward)
        plt.title("Mean Reward (Last 100 Episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        if out_prefix is not None:
            plt.savefig(f"{out_prefix}_mean_reward_100.png", dpi=200)
        plt.close()

        plt.figure(figsize=figsize)
        plt.plot(x_vals, steps)
        plt.title("Mean Steps to Goal (Last 100 Episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.tight_layout()
        if out_prefix is not None:
            plt.savefig(f"{out_prefix}_mean_steps_100.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------
# Epsilon-greedy action selection
# ---------------------------------------------------------------------
def epsilon_greedy_action(env, q_values, eps):
    if random.random() > eps:
        return int(np.argmax(q_values))
    return env.action_space.sample()


# ---------------------------------------------------------------------
# Main Q-learning loop
# ---------------------------------------------------------------------
def Q_Learning(
    writer,
    lr,
    discount_factor,
    max_epsilon,
    decay_factor,
    linear=True,
    plot=True,
    num_episodes=5000,
    max_steps_per_episode=100,
    run_name="run",
):
    np.random.seed(1)
    random.seed(1)

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=1)
    env.action_space.seed(1)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards_episode = []
    avg_steps_list = []
    avg_rewards_list = []
    steps_all = []

    eps_schedule = get_glie(num_episodes, max_epsilon, decay_factor, linear)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = int(state)
        eps = eps_schedule[episode]
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps_per_episode):
            steps += 1
            action = epsilon_greedy_action(env, q_table[state, :], eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = int(next_state)

            # Q-learning update
            best_next_q = np.max(q_table[next_state, :])
            td_target = reward + discount_factor * best_next_q * (0.0 if done else 1.0)
            q_table[state, action] = (1 - lr) * q_table[state, action] + lr * td_target

            state = next_state
            total_reward += reward

            if done:
                if reward < 1.0:
                    steps = max_steps_per_episode
                break

        steps_all.append(steps)
        rewards_episode.append(total_reward)

        if writer is not None:
            writer.add_scalar("Q1/Reward_per_episode", total_reward, episode)
            writer.add_scalar("Q1/Steps_to_goal", steps, episode)

        # Every 100 episodes compute rolling averages
        if (episode + 1) % 100 == 0:
            mean_steps = float(np.mean(steps_all[-100:]))
            mean_reward = float(np.mean(rewards_episode[-100:]))
            avg_steps_list.append(mean_steps)
            avg_rewards_list.append(mean_reward)

            if writer is not None:
                writer.add_scalar("Q1/Mean_reward_100", mean_reward, episode + 1)
                writer.add_scalar("Q1/Mean_steps_100", mean_steps, episode + 1)

            print(
                f"[{run_name}] Ep {episode+1}: avgR={mean_reward:.3f}, "
                f"avgSteps={mean_steps:.1f}, eps={eps:.3f}"
            )

        # Save intermediate Q-tables at 500 and 2000 episodes
        if episode + 1 in (500, 2000):
            prefix = os.path.join(FIG_DIR, f"{run_name}_ep{episode+1}")
            plot_graphs(q_table, f"Q-table after {episode+1} episodes",
                        heat=True, out_prefix=prefix)

    # Final plots
    if plot:
        final_prefix = os.path.join(FIG_DIR, f"{run_name}_final")
        plot_graphs(
            q=q_table,
            title="Final Q-table",
            rewards=rewards_episode,
            avg_reward=avg_rewards_list,
            steps=avg_steps_list,
            out_prefix=final_prefix,
        )

    env.close()
    return q_table, rewards_episode, avg_steps_list, avg_rewards_list


# ---------------------------------------------------------------------
# Hyperparameter sweep (for the figure in Section 1.5)
# ---------------------------------------------------------------------
def Optimize_Q_Learning(num_episodes_sweep=3000):
    params = OrderedDict(
        learning_rate=[0.1, 0.01, 0.001],
        discount_factor=[0.99, 0.999],
        linear=[True, False],
        decay_factor=[0.99, 0.9995],
    )

    plt.figure(figsize=(10, 6))

    for run in RunBuilder.get_runs(params):
        run_name = (
            f"lr={run.learning_rate}_g={run.discount_factor}_"
            f"{'lin' if run.linear else 'exp'}_d={run.decay_factor}"
        )
        writer = SummaryWriter(log_dir=f"runs/q1_sweep/{run_name}")

        _, _, _, avg_reward_100 = Q_Learning(
            writer=writer,
            lr=run.learning_rate,
            discount_factor=run.discount_factor,
            max_epsilon=1.0,
            decay_factor=run.decay_factor,
            linear=run.linear,
            plot=False,
            num_episodes=num_episodes_sweep,
            run_name=run_name,
        )
        writer.close()

        x_axis = np.arange(len(avg_reward_100)) * 100
        plt.plot(x_axis, avg_reward_100, label=run_name)

    plt.title("Q1: Mean Reward (Last 100 Episodes) – Hyperparameter Sweep")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "q1_hyperparam_sweep.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved sweep figure to {path}")


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Final chosen hyperparameters (based on sweep)
    learning_rate = 0.1
    discount_factor = 0.999
    linear = True
    decay_factor = 0.999

    writer = SummaryWriter(log_dir="runs/q1_final")

    Q_Learning(
        writer=writer,
        lr=learning_rate,
        discount_factor=discount_factor,
        max_epsilon=1.0,
        decay_factor=decay_factor,
        linear=linear,
        plot=True,
        run_name="q1",
    )
    writer.close()

    # Perform sweep (optional)
    Optimize_Q_Learning(num_episodes_sweep=3000)
