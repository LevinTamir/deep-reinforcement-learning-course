# ============================================================
#  Section 1 – Training Individual Networks (MountainCarContinuous-v0)
#  Actor-Critic (TD(0)) with HW-compatible fixed input/output sizes
#
#  Key fixes vs your current file:
#   1) Keep TRUE env reward separate from shaped reward (no "fake" 500 returns)
#   2) Greedy evaluation uses tanh(mu) (avoids constant +/-1 action, eval ~ -25)
#   3) Shaping (optional) is used ONLY for learning targets/advantages
#   4) Solve criterion is based on greedy evaluation on TRUE env reward
# ============================================================

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config
# -----------------------------
@dataclass
class A2CConfig:
    env_name: str = "MountainCarContinuous-v0"

    # RL
    gamma: float = 0.99
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    hidden: int = 256
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    normalize_advantages: bool = False
    max_grad_norm: float = 0.5

    # Training
    max_episodes: int = 300
    seed: int = 123
    eval_every: int = 10
    eval_episodes: int = 10
    solve_score: float = 90.0  # greedy-eval threshold (TRUE env return)

    # Fixed IO sizes (HW requirement)
    fixed_obs_dim: int = 6
    fixed_actor_out_dim: int = 3

    # Gaussian policy stabilization
    log_std_min: float = -3.0
    log_std_max: float = 1.0

    # Optional: per-step discount accumulator I_t (as in TF ref)
    use_I_weight: bool = True

    # Reward shaping toggle (learning only)
    use_reward_shaping: bool = True

    # Plotting
    plot_path: str = "plots/mountaincarcontinuous_actor_critic.png"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_observation(obs: np.ndarray, target_size: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[0] < target_size:
        obs = np.pad(obs, (0, target_size - obs.shape[0]), mode="constant", constant_values=0.0)
    return obs[:target_size]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def moving_average(x: list[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(train_returns_env, eval_returns, eval_every, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(train_returns_env) + 1)
    eval_x = np.arange(eval_every, eval_every * len(eval_returns) + 1, eval_every)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episodes, train_returns_env, alpha=0.35, label="Train return (env)")
    ma = moving_average(train_returns_env, 20)
    if len(ma) > 1:
        ma_x = np.arange(20, 20 + len(ma))
        plt.plot(ma_x, ma, linewidth=2, label="Train return (env, MA20)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel("Return")

    plt.subplot(2, 1, 2)
    if len(eval_returns) > 0:
        plt.plot(eval_x, eval_returns, marker="o", linewidth=1.5, label="Greedy eval avg (env)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


# -----------------------------
# Networks
# -----------------------------
class Actor(nn.Module):
    """
    Fixed output size = 3:
      out[0] -> mu (mean)
      out[1] -> log_std
      out[2] -> dummy (unused)
    """
    def __init__(self, obs_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, out_dim), std=0.01)

        # Mild bias to avoid "do nothing" optimum (keep small)
        with torch.no_grad():
            self.fc3.bias[0].fill_(0.2)   # mu bias (small push)
            self.fc3.bias[1].fill_(-0.5)  # log_std bias (moderate std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


# -----------------------------
# Policy helpers
# -----------------------------
def sample_action_and_logprob(cfg: A2CConfig, actor_out: torch.Tensor):
    """
    Stochastic policy for training:
      - Sample a ~ Normal(mu, std)
      - Use tanh(a) to bound into [-1, 1] (action actually sent to env)
      - Use log_prob with tanh correction for consistency
    """
    mu = actor_out[0]
    log_std = actor_out[1].clamp(cfg.log_std_min, cfg.log_std_max)
    std = torch.exp(log_std)

    dist = torch.distributions.Normal(mu, std)
    pre_tanh = dist.rsample()                 # scalar
    action = torch.tanh(pre_tanh)             # scalar in [-1, 1]

    # log pi(a) with tanh correction: log p(u) - log(1 - tanh(u)^2)
    log_prob_u = dist.log_prob(pre_tanh)
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = (log_prob_u - correction).squeeze()
    entropy = dist.entropy().squeeze()

    return action.unsqueeze(0), log_prob, entropy


@torch.no_grad()
def evaluate_policy_greedy(cfg: A2CConfig, actor: Actor, scaler: StandardScaler, episodes: int) -> float:
    """
    Greedy evaluation on TRUE env reward:
      a = tanh(mu)
    """
    env = gym.make(cfg.env_name)
    returns = []

    actor_was_training = actor.training
    actor.eval()

    for i in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + 10_000 + i)
        done = False
        ep_ret_env = 0.0

        while not done:
            obs_norm = scaler.transform(np.asarray(obs, dtype=np.float32).reshape(1, -1)).flatten()
            obs_pad = pad_observation(obs_norm, cfg.fixed_obs_dim)
            obs_t = torch.as_tensor(obs_pad, dtype=torch.float32)

            out = actor(obs_t)
            mu = out[0]
            action = torch.tanh(mu)  # key change: consistent bounded mean action
            action_np = action.cpu().numpy().astype(np.float32).reshape(1,)

            obs, r_env, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            ep_ret_env += float(r_env)

        returns.append(ep_ret_env)

    env.close()
    if actor_was_training:
        actor.train()
    return float(np.mean(returns))


# -----------------------------
# Reward shaping (optional, learning only)
# -----------------------------
def shaped_reward_energy(obs: np.ndarray, obs2: np.ndarray, r_env: float) -> float:
    """
    Energy-based shaping used ONLY for learning (targets/advantages), not for reporting.
    """
    height_old = np.sin(3 * obs[0])
    height_new = np.sin(3 * obs2[0])

    kinetic_old = 0.5 * obs[1] ** 2
    kinetic_new = 0.5 * obs2[1] ** 2

    energy_bonus = 100.0 * ((height_new + 50 * kinetic_new) - (height_old + 50 * kinetic_old))

    if obs2[0] >= 0.45:
        energy_bonus += 100.0
    elif obs2[0] > 0.3:
        energy_bonus += 10.0

    return float(r_env) + float(energy_bonus)


# -----------------------------
# Training
# -----------------------------
def train(cfg: A2CConfig):
    set_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    obs0, _ = env.reset(seed=cfg.seed)

    # ---- Fit scaler on actual trajectories ----
    print("Collecting initial trajectories for scaler...")
    scaler_obs = []
    for _ in range(50):  # 50 random episodes
        o, _ = env.reset()
        for _ in range(200):
            scaler_obs.append(o.copy())
            a = env.action_space.sample()
            o, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    scaler = StandardScaler()
    scaler.fit(np.array(scaler_obs, dtype=np.float32))
    print(f"Scaler fitted on {len(scaler_obs)} observations")

    actor = Actor(cfg.fixed_obs_dim, cfg.fixed_actor_out_dim, cfg.hidden)
    critic = Critic(cfg.fixed_obs_dim, cfg.hidden)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    print(f"Environment: {cfg.env_name}")
    print(f"Original observation size: {np.asarray(obs0).shape[0]}")
    print(f"Padded observation size (input): {cfg.fixed_obs_dim}")
    print(f"Network output size (fixed): {cfg.fixed_actor_out_dim}")
    print("Action bounds: [-1.0, 1.0]")
    print(f"Hidden layer size: {cfg.hidden}")

    train_returns_env = []
    eval_returns = []

    best_greedy = -1e9
    best_actor_state = None
    solved_ep = -1

    start_time = time.time()

    for ep in range(cfg.max_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False

        ep_ret_env = 0.0      # TRUE env return (for reporting)
        ep_ret_shaped = 0.0   # shaped return (for debugging)

        log_probs = []
        entropies = []
        values = []
        targets = []
        advantages = []
        I_weights = []

        I = 1.0

        while not done:
            # Normalize + pad
            obs_norm = scaler.transform(np.asarray(obs, dtype=np.float32).reshape(1, -1)).flatten()
            obs_pad = pad_observation(obs_norm, cfg.fixed_obs_dim)
            obs_t = torch.as_tensor(obs_pad, dtype=torch.float32)

            out = actor(obs_t)
            action_t, log_prob, entropy = sample_action_and_logprob(cfg, out)
            value = critic(obs_t)

            # Step env (detach!)
            action_np = action_t.detach().cpu().numpy().astype(np.float32)
            obs2, r_env, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)

            # Track TRUE env return
            r_env_f = float(r_env)
            ep_ret_env += r_env_f

            # Learning reward (optionally shaped)
            if cfg.use_reward_shaping:
                r_learn = shaped_reward_energy(obs, obs2, r_env_f)
            else:
                r_learn = r_env_f

            ep_ret_shaped += float(r_learn)

            # Critic bootstrap
            with torch.no_grad():
                obs2_norm = scaler.transform(np.asarray(obs2, dtype=np.float32).reshape(1, -1)).flatten()
                obs2_pad = pad_observation(obs2_norm, cfg.fixed_obs_dim)
                obs2_t = torch.as_tensor(obs2_pad, dtype=torch.float32)
                next_value = critic(obs2_t)
                bootstrap = 0.0 if done else next_value.item()
                td_target = float(r_learn) + cfg.gamma * bootstrap
                td_error = td_target - value.item()

            # Store
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            targets.append(td_target)
            advantages.append(td_error)
            I_weights.append(I)

            obs = obs2
            I *= cfg.gamma

        # Convert to tensors
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        values_t = torch.stack(values)
        targets_t = torch.tensor(targets, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)

        if cfg.normalize_advantages and len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        if cfg.use_I_weight:
            I_t = torch.tensor(I_weights, dtype=torch.float32)
        else:
            I_t = torch.ones_like(adv_t)

        # Losses
        actor_loss = -(log_probs_t * adv_t * I_t).mean()
        entropy_bonus = cfg.entropy_coef * entropies_t.mean()
        critic_loss = cfg.value_loss_coef * ((targets_t - values_t) ** 2).mean()

        total_loss = actor_loss + critic_loss - entropy_bonus

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        total_loss.backward()

        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)

        opt_actor.step()
        opt_critic.step()

        # Report TRUE env return
        train_returns_env.append(ep_ret_env)

        # Greedy evaluation on TRUE env return
        if (ep + 1) % cfg.eval_every == 0:
            greedy = evaluate_policy_greedy(cfg, actor, scaler, cfg.eval_episodes)
            eval_returns.append(greedy)

            if greedy > best_greedy:
                best_greedy = greedy
                best_actor_state = deepcopy(actor.state_dict())

            avg100 = float(np.mean(train_returns_env[-100:])) if len(train_returns_env) >= 100 else float("nan")

            print(
                f"Episode {ep+1:4d}/{cfg.max_episodes} | "
                f"train_env_return={ep_ret_env:8.2f} | "
                f"train_learn_return={ep_ret_shaped:8.2f} | "
                f"avg100_env={avg100:8.2f} | "
                f"eval_avg_env={greedy:8.2f}"
            )

            if best_greedy >= cfg.solve_score:
                solved_ep = ep + 1
                print(f"SOLVED at episode {solved_ep} with best_greedy={best_greedy:.1f}")
                break

    env.close()

    # Restore best actor
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)

    final_eval = evaluate_policy_greedy(cfg, actor, scaler, cfg.eval_episodes)
    elapsed = time.time() - start_time

    print(f"Final greedy eval (env, {cfg.eval_episodes} eps): {final_eval:.2f}")
    print(f"Best greedy eval (env): {best_greedy:.2f}")
    print(f"Solved episode: {solved_ep if solved_ep != -1 else 'Not solved'}")
    print(f"Elapsed time: {elapsed:.2f}s")

    # Save plot + results (TRUE env returns)
    plot_learning_curves(train_returns_env, eval_returns, cfg.eval_every, cfg.plot_path)

    Path("results").mkdir(parents=True, exist_ok=True)
    np.savez(
        "results/mountaincarcontinuous_actor_critic.npz",
        train_env_returns=np.array(train_returns_env, dtype=np.float32),
        eval_env_returns=np.array(eval_returns, dtype=np.float32),
        eval_every=cfg.eval_every,
        solved_ep=solved_ep,
        best_greedy=best_greedy,
        final_eval=final_eval,
        elapsed=elapsed,
        used_reward_shaping=cfg.use_reward_shaping,
    )


def main():
    cfg = A2CConfig(
        env_name="MountainCarContinuous-v0",
        gamma=0.99,
        lr_actor=1e-4,
        lr_critic=5e-4,
        hidden=256,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        normalize_advantages=False,
        max_grad_norm=0.5,
        max_episodes=300,
        seed=123,
        eval_every=10,
        eval_episodes=10,
        solve_score=90.0,
        fixed_obs_dim=6,
        fixed_actor_out_dim=3,
        log_std_min=-3.0,
        log_std_max=1.0,
        use_I_weight=True,
        use_reward_shaping=True,  # set False if you must not shape rewards for HW
        plot_path="plots/mountaincarcontinuous_actor_critic.png",
    )

    print("\n=== Hyperparameters ===")
    print(f"gamma: {cfg.gamma}")
    print(f"lr_actor: {cfg.lr_actor}")
    print(f"lr_critic: {cfg.lr_critic}")
    print(f"entropy_coef: {cfg.entropy_coef}")
    print(f"value_loss_coef: {cfg.value_loss_coef}")
    print(f"max_grad_norm: {cfg.max_grad_norm}")
    print(f"max_episodes: {cfg.max_episodes}")
    print(f"solve_score: {cfg.solve_score}")
    print("=======================\n")

    train(cfg)


if __name__ == "__main__":
    main()
