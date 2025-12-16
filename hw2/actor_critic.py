# actor_critic.py
"""
TD-Based Actor-Critic for CartPole-v1 (HW2 Section 2)

Key difference from REINFORCE with Baseline:
- Uses TD-error δ = r + γV(s') - V(s) as advantage estimate (not Monte-Carlo returns)
- Updates are batched at episode end for stability (still TD-based advantages)
- Normalizes advantages for stable training
"""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

# -----------------------------
# Config
# -----------------------------
@dataclass
class A2CConfig:
    env_name: str = "CartPole-v1"

    gamma: float = 0.99
    lr_actor: float = 3e-4       # Adam-friendly learning rate
    lr_critic: float = 1e-3      # Critic can learn faster
    hidden: int = 128

    # Stabilizers
    entropy_coef: float = 0.01   # Mild entropy bonus for exploration
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5   # Gradient clipping
    normalize_advantages: bool = True  # Critical for stable training

    max_episodes: int = 2000
    seed: int = 543

    print_every: int = 10
    eval_every: int = 50
    eval_episodes: int = 10
    solve_score: float = 475.0


# -----------------------------
# Models
# -----------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for weights, constant for bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Policy network π(a|s; θ)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # Small std for near-uniform initial policy
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """Value function V(s; w)."""
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_name: str):
    try:
        import gymnasium as gym
        return gym.make(env_name), "gymnasium"
    except Exception:
        import gym
        return gym.make(env_name), "gym"


def reset_env(env, api: str, seed: Optional[int] = None):
    if api == "gymnasium":
        obs, _info = env.reset(seed=seed)
        return obs
    else:
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                pass
        obs = env.reset()
        return obs


def step_env(env, api: str, action: int):
    if api == "gymnasium":
        obs2, reward, terminated, truncated, info = env.step(action)
        return obs2, float(reward), bool(terminated), bool(truncated), info
    else:
        obs2, reward, done, info = env.step(action)
        return obs2, float(reward), bool(done), False, info


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize tensor to zero mean / unit std."""
    return (x - x.mean()) / (x.std() + eps)


def moving_average(x: List[float], window: int) -> np.ndarray:
    """Simple moving average using convolution; returns an array shorter by window-1."""
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(
    train_returns: List[float],
    avg100_returns: List[float],
    eval_returns: List[float],
    eval_every: int,
    out_path: str = "plots/actor_critic_learning_curve.png",
    ma_window: int = 50,
    title: str = "Actor-Critic learning curve",
) -> None:
    """
    Plots:
      - Reward per episode
      - Average reward over last 100 episodes
      - (Optional) moving average of reward per episode
      - Eval average reward every eval_every episodes
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(train_returns) + 1)
    eval_episodes = np.arange(eval_every, eval_every * len(eval_returns) + 1, eval_every)

    plt.figure(figsize=(12, 6))

    # --- Subplot 1: Reward per episode ---
    plt.subplot(2, 1, 1)
    plt.plot(episodes, train_returns, alpha=0.4, label="Reward per episode")
    ma = moving_average(train_returns, ma_window)
    if len(ma) > 1:
        ma_x = np.arange(ma_window, ma_window + len(ma))
        plt.plot(ma_x, ma, linewidth=2, label=f"Reward (MA{ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Subplot 2: Avg reward over last 100 episodes + eval ---
    plt.subplot(2, 1, 2)
    plt.plot(episodes, avg100_returns, linewidth=2, label="Avg reward (last 100 episodes)")
    if len(eval_returns) > 0:
        plt.plot(eval_episodes, eval_returns, marker="o", linewidth=1.5, label="Eval avg return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved learning curves to {out_path}")


def compute_td_advantages(
    rewards: List[float],
    values: List[torch.Tensor],
    next_values: List[torch.Tensor],
    dones: List[bool],
    gamma: float
) -> torch.Tensor:
    """
    Compute TD(0) advantages: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
    
    This is the key difference from REINFORCE with Baseline:
    - REINFORCE uses Monte-Carlo returns: A_t = G_t - V(s_t)
    - Actor-Critic uses TD-error: δ_t = r + γV(s') - V(s)
    """
    advantages = []
    for r, v, v_next, done in zip(rewards, values, next_values, dones):
        # TD-error: δ = r + γV(s') - V(s)
        # If done (terminated), V(s') = 0
        bootstrap = 0.0 if done else v_next.item()
        td_target = r + gamma * bootstrap
        td_error = td_target - v.item()
        advantages.append(td_error)
    return torch.tensor(advantages, dtype=torch.float32)


@torch.no_grad()
def evaluate_policy(cfg: A2CConfig, actor: Actor, episodes: int = 10) -> float:
    """Evaluate policy using greedy (argmax) action selection."""
    was_training = actor.training
    actor.eval()
    env, api = make_env(cfg.env_name)
    returns = []
    for i in range(episodes):
        obs = reset_env(env, api, seed=cfg.seed + 10000 + i)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits = actor(obs_t)
            action = int(torch.argmax(logits).item())
            obs, r, terminated, truncated, _ = step_env(env, api, action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    if was_training:
        actor.train()
    return float(np.mean(returns))


# -----------------------------
# Training - TD-Based Actor-Critic
# -----------------------------
def train_a2c(cfg: A2CConfig) -> Tuple[List[float], int, float]:
    set_seed(cfg.seed)

    env, api = make_env(cfg.env_name)
    obs = reset_env(env, api, seed=cfg.seed)

    obs_dim = int(np.asarray(obs).shape[0])
    act_dim = int(env.action_space.n)

    actor = Actor(obs_dim, act_dim, cfg.hidden)
    critic = Critic(obs_dim, cfg.hidden)

    # Adam optimizer - more stable than RMSprop for this setup
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    episode_returns: List[float] = []
    avg100_returns: List[float] = []  # Track avg100 for plotting
    eval_returns: List[float] = []    # Track eval returns for plotting
    solved_ep = -1
    best_greedy = -1e9
    best_actor_state = None
    last_greedy = float("nan")

    for ep in range(cfg.max_episodes):
        obs = reset_env(env, api, seed=cfg.seed + ep)
        done = False

        # Collect episode trajectory
        states = []
        actions = []
        log_probs = []
        entropies = []
        values = []
        next_values = []
        rewards = []
        dones = []  # terminated flags (for bootstrapping)

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            
            # Actor: sample action
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()
            
            # Critic: estimate value
            value = critic(obs_t)
            
            # Environment step
            obs2, reward, terminated, truncated, _ = step_env(env, api, action)
            done = terminated or truncated
            
            # Next state value (for TD target)
            with torch.no_grad():
                obs2_t = torch.as_tensor(obs2, dtype=torch.float32)
                next_value = critic(obs2_t)
            
            # Store transition
            states.append(obs_t)
            actions.append(action_t)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            dones.append(terminated)  # Use terminated, not truncated, for bootstrap decision
            
            obs = obs2

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)

        # Stack tensors
        log_probs_t = torch.stack(log_probs)    # [T]
        entropies_t = torch.stack(entropies)    # [T]
        values_t = torch.stack(values)          # [T]
        
        # Compute TD advantages: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        # This is the key Actor-Critic formulation (different from Monte-Carlo)
        advantages = compute_td_advantages(rewards, values, next_values, dones, cfg.gamma)
        
        # Normalize advantages for stable training
        if cfg.normalize_advantages and len(advantages) > 1:
            advantages = normalize(advantages)

        # === Actor Loss ===
        # L_actor = -E[log π(a|s) * δ] - entropy_coef * H(π)
        actor_loss = -(log_probs_t * advantages).mean()
        entropy_loss = -cfg.entropy_coef * entropies_t.mean()

        # === Critic Loss ===
        # L_critic = E[(r + γV(s') - V(s))^2] = E[δ^2]
        # Recompute TD targets for gradient computation
        td_targets = []
        for r, v_next, terminal in zip(rewards, next_values, dones):
            bootstrap = 0.0 if terminal else v_next.item()
            td_targets.append(r + cfg.gamma * bootstrap)
        td_targets_t = torch.tensor(td_targets, dtype=torch.float32)
        
        critic_loss = cfg.value_loss_coef * ((td_targets_t - values_t) ** 2).mean()

        # === Total Loss & Optimization ===
        total_loss = actor_loss + critic_loss + entropy_loss

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        total_loss.backward()
        
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        
        opt_actor.step()
        opt_critic.step()

        avg100 = float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 100 else float("nan")
        avg100_returns.append(avg100)  # Track for plotting

        # Periodic evaluation
        if (ep + 1) % cfg.eval_every == 0:
            last_greedy = evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)
            eval_returns.append(last_greedy)  # Track for plotting
            if last_greedy > best_greedy:
                best_greedy = last_greedy
                best_actor_state = deepcopy(actor.state_dict())

        if (ep + 1) % cfg.print_every == 0:
            if len(episode_returns) >= 100:
                print(f"ep={ep+1:4d}  ret={ep_ret:7.1f}  avg100={avg100:7.2f}  greedy={last_greedy:7.1f}")
            else:
                print(f"ep={ep+1:4d}  ret={ep_ret:7.1f}  greedy={last_greedy:7.1f}")

        # Check solve condition
        if best_greedy >= cfg.solve_score:
            solved_ep = ep + 1
            print(f"SOLVED at episode {solved_ep} with best_greedy={best_greedy:.1f}")
            break

    env.close()
    
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
    
    final_eval = evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)
    print(f"Final greedy eval ({cfg.eval_episodes} eps): {final_eval:.1f}")
    
    # Plot learning curves
    plot_learning_curves(
        train_returns=episode_returns,
        avg100_returns=avg100_returns,
        eval_returns=eval_returns,
        eval_every=cfg.eval_every,
        out_path="plots/actor_critic_learning_curve.png",
        ma_window=50,
        title=f"Actor-Critic (TD-based) - {cfg.env_name}",
    )
    
    return episode_returns, solved_ep, best_greedy


def main():
    cfg = A2CConfig(
        env_name="CartPole-v1",
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=1e-3,
        hidden=128,
        
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=True,
        
        max_episodes=2000,
        seed=543,
        print_every=10,
        eval_every=50,
        eval_episodes=10,
        solve_score=475.0,
    )
    returns, solved_ep, best_greedy = train_a2c(cfg)
    print(f"Done. episodes={len(returns)}, solved_ep={solved_ep}, best_greedy={best_greedy:.1f}")


if __name__ == "__main__":
    main()
