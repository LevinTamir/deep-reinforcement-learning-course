# ============================================
# HW3 Plotting Utils
# Mark Feldman (320827637) & Tamir Levin (315765347)
# ============================================


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def moving_average(x: list[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(
    train_returns: list[float],
    avg100_returns: list[float],
    eval_returns: list[float],
    eval_every: int,
    out_path: str = "plots/actor_critic_learning_curve.png",
    ma_window: int = 50,
    title: str = "Actor-Critic learning curve",
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(train_returns) + 1)
    eval_episodes = np.arange(eval_every, eval_every * len(eval_returns) + 1, eval_every)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, train_returns, alpha=0.4, label="Reward per episode")
    ma = moving_average(train_returns, ma_window)
    if len(ma) > 1:
        ma_x = np.arange(ma_window, ma_window + len(ma))
        plt.plot(ma_x, ma, linewidth=2, label=f"Reward (MA{ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, avg100_returns, linewidth=2, label="Avg reward (last 100 episodes)")
    if len(eval_returns) > 0:
        plt.plot(eval_episodes, eval_returns, marker="o", linewidth=1.5, label="Eval avg return")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved learning curves to {out_path}")