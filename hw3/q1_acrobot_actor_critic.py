# ============================================
# Section 1 – Training Individual Networks (Acrobot-v1)
# ============================================

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym

@dataclass
class A2CConfig:
    env_name: str = "Acrobot-v1"

    gamma: float = 0.99
    lr_actor: float = 2e-3
    lr_critic: float = 1e-3
    hidden: int = 256

    #learning rate decay
    lr_step_size: int = 40
    lr_gamma: float = 0.7
    min_lr: float = 1e-4

    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    max_episodes: int = 2000
    seed: int = 543

    print_every: int = 10
    eval_every: int = 25
    eval_episodes: int = 10
    solve_score: float = -100.0  # Acrobot: solved when avg return >= -100


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def pad_observation(obs: np.ndarray, target_size: int = 6) -> np.ndarray:
    """Pad observation to target size with zeros for meta-learning compatibility."""
    obs = np.asarray(obs, dtype=np.float32)
    if len(obs) < target_size:
        obs = np.pad(obs, (0, target_size - len(obs)), mode='constant', constant_values=0.0)
    return obs[:target_size]


class Actor(nn.Module):
    """policy network π(a|s; θ) - ReLU/ELU architecture matching reference"""
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.obs_dim = 6
        self.act_dim = 3
        self.fc1 = layer_init(nn.Linear(self.obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, self.act_dim), std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    """value function V(s; w) - ReLU/ELU architecture matching reference"""
    def __init__(self, obs_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.obs_dim = 6
        self.fc1 = layer_init(nn.Linear(self.obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def step_env(env, api: str, action: int):
    obs2, reward, terminated, truncated, info = env.step(action)
    return obs2, float(reward), bool(terminated), bool(truncated), info

def moving_average(x: list[float], window: int) -> np.ndarray:
    """simple moving average"""
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


def compute_td_advantages(
    rewards: list[float],
    values: list[torch.Tensor],
    next_values: list[torch.Tensor],
    dones: list[bool],
    gamma: float
) -> torch.Tensor:
    """
    compute TD(0) advantages: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
    
    the difference from REINFORCE with Baseline:
    - reinforce uses monte-carlo returns: A_t = G_t - V(s_t)
    - actor-critic uses td-error: δ_t = r + γV(s') - V(s)
    """

    advantages = []
    for r, v, v_next, done in zip(rewards, values, next_values, dones):
        # td-error: δ = r + γV(s') - V(s)
        bootstrap = 0.0 if done else v_next.item()
        td_target = r + gamma * bootstrap
        td_error = td_target - v.item()
        advantages.append(td_error)
    return torch.tensor(advantages, dtype=torch.float32)


@torch.no_grad()
def evaluate_policy(cfg: A2CConfig, actor: Actor, episodes: int = 10) -> float:
    """evaluate policy using argmax action selection"""
    was_training = actor.training
    actor.eval()
    env = gym.make(cfg.env_name)
    api = "gymnasium"

    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + 10000 + i)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(pad_observation(obs), dtype=torch.float32)
            logits = actor(obs_t)
            action = int(torch.argmax(logits).item()) % env.action_space.n
            obs, r, terminated, truncated, _ = step_env(env, api, action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    if was_training:
        actor.train()
    return float(np.mean(returns))

def train_a2c(cfg: A2CConfig) -> tuple[list[float], int, float, float, int]:
    """Train actor-critic agent and return results with timing statistics.
    
    Returns:
        episode_returns: list of returns per episode
        solved_ep: episode at which solve threshold was reached (-1 if not solved)
        best_greedy: best greedy evaluation return
        elapsed_time: total training time in seconds
        num_iterations: total number of training iterations
    """
    start_time = time.time()
    set_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    api =  "gymnasium"
    obs, _ = env.reset(seed=cfg.seed)

    obs_dim = int(np.asarray(obs).shape[0])
    act_dim = int(env.action_space.n)
    
    # Use fixed 6 input, 3 output for meta-learning compatibility
    fixed_obs_dim = 6
    fixed_act_dim = 3
    
    print(f"Environment: {cfg.env_name}")
    print(f"Original observation size: {obs_dim}")
    print(f"Padded observation size (input): {fixed_obs_dim}")
    print(f"Original action size: {act_dim}")
    print(f"Network output size: {fixed_act_dim}")
    print(f"Hidden layer size: {cfg.hidden}")

    actor = Actor(fixed_obs_dim, fixed_act_dim, cfg.hidden)
    critic = Critic(fixed_obs_dim, cfg.hidden)

    # adam optimizer
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # learning rate schedulers
    scheduler_actor = StepLR(opt_actor, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    scheduler_critic = StepLR(opt_critic, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    episode_returns = []
    avg100_returns = []
    eval_returns = []  
    solved_ep = -1
    best_greedy = -1e9
    best_actor_state = None
    last_greedy = float("nan")
    total_iterations = 0

    for ep in range(cfg.max_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        
        states = []
        actions = []
        log_probs = []
        entropies = []
        values = []
        next_values = []
        rewards = []
        dones = [] 

        while not done:
            obs_t = torch.as_tensor(pad_observation(obs), dtype=torch.float32)
            # actor: sample action
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item()) % act_dim
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()
            
            # critic: estimate value
            value = critic(obs_t)
            
            # environment step
            obs2, reward, terminated, truncated, _ = step_env(env, api, action)
            done = terminated or truncated
            
            # next state value
            with torch.no_grad():
                obs2_t = torch.as_tensor(pad_observation(obs2), dtype=torch.float32)
                next_value = critic(obs2_t)
            
            # store transition
            states.append(obs_t)
            actions.append(action_t)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            dones.append(terminated)
            
            obs = obs2

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)
        total_iterations += len(rewards)  # Count total training steps

        log_probs_t = torch.stack(log_probs) 
        entropies_t = torch.stack(entropies) 
        values_t = torch.stack(values)     

        advantages = compute_td_advantages(rewards, values, next_values, dones, cfg.gamma)
        
        if cfg.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        actor_loss = -(log_probs_t * advantages).mean()
        entropy_loss = -cfg.entropy_coef * entropies_t.mean()

        td_targets = []
        for r, v_next, terminal in zip(rewards, next_values, dones):
            bootstrap = 0.0 if terminal else v_next.item()
            td_targets.append(r + cfg.gamma * bootstrap)
        td_targets_t = torch.tensor(td_targets, dtype=torch.float32)
        
        critic_loss = cfg.value_loss_coef * ((td_targets_t - values_t) ** 2).mean()

        total_loss = actor_loss + critic_loss + entropy_loss

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        total_loss.backward()
        
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        
        opt_actor.step()
        opt_critic.step()

        scheduler_actor.step()
        scheduler_critic.step()

        for param_group in opt_actor.param_groups:
            param_group['lr'] = max(param_group['lr'], cfg.min_lr)
        for param_group in opt_critic.param_groups:
            param_group['lr'] = max(param_group['lr'], cfg.min_lr)

        avg100 = float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 100 else float("nan")
        avg100_returns.append(avg100) 

        if (ep + 1) % cfg.eval_every == 0:
            last_greedy = evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)
            eval_returns.append(last_greedy) 
            if last_greedy > best_greedy:
                best_greedy = last_greedy
                best_actor_state = deepcopy(actor.state_dict())
            
            print(
                f"Episode {ep+1:5d} | "
                f"train_return={ep_ret:8.2f} | "
                f"avg100={avg100:8.2f} | "
                f"eval_avg={last_greedy:8.2f}"
            )

        if best_greedy >= cfg.solve_score:
            solved_ep = ep + 1
            print(f"SOLVED at episode {solved_ep} with best_greedy={best_greedy:.1f}")
            break

    env.close()
    elapsed_time = time.time() - start_time
    
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
    
    final_eval = evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)
    print(f"Final greedy eval ({cfg.eval_episodes} eps): {final_eval:.1f}")
    print(f"\n--- Training Statistics ---")
    print(f"Total episodes: {ep + 1}")
    print(f"Total training iterations: {total_iterations}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Solved at episode: {solved_ep if solved_ep != -1 else 'Not solved'}")
    print(f"Best greedy return: {best_greedy:.2f}")
    
    plot_learning_curves(
        train_returns=episode_returns,
        avg100_returns=avg100_returns,
        eval_returns=eval_returns,
        eval_every=cfg.eval_every,
        out_path="plots/acrobot_actor_critic.png",
        ma_window=50,
        title=f"Actor-Critic - {cfg.env_name}",
    )
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "acrobot_actor_critic.npz",
        train_returns=np.array(episode_returns),
        avg100_returns=np.array(avg100_returns),
        eval_returns=np.array(eval_returns),
        eval_every=cfg.eval_every,
        solved_ep=solved_ep if solved_ep != -1 else -1,
        best_greedy=best_greedy,
        elapsed_time=elapsed_time,
        total_iterations=total_iterations,
    )
    
    # Save model for transfer learning (Section 2)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), models_dir / "acrobot_actor.pt")
    torch.save(critic.state_dict(), models_dir / "acrobot_critic.pt")
    print(f"Saved models to {models_dir}/")
    
    return episode_returns, solved_ep, best_greedy, elapsed_time, total_iterations


def main():
    cfg = A2CConfig(
        env_name="Acrobot-v1",
        gamma=0.99,
        
        lr_actor=1e-3,
        lr_critic=5e-3,
        lr_step_size=100,
        lr_gamma=0.95,
        min_lr=1e-4,
        
        hidden=256,
        
        entropy_coef=0.05,
        value_loss_coef=0.25,
        max_grad_norm=1.0,
        normalize_advantages=True,
        
        max_episodes=2000,
        seed=123,                  
        print_every=10,
        eval_every=25,
        eval_episodes=10,
        solve_score=-100.0,  # Acrobot: solved when avg return >= -100
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
    returns, solved_ep, best_greedy, elapsed_time, total_iterations = train_a2c(cfg)
    print(f"Done. episodes={len(returns)}, solved_ep={solved_ep}, best_greedy={best_greedy:.1f}")
    print(f"Elapsed time: {elapsed_time:.2f}s, Total iterations: {total_iterations}")


if __name__ == "__main__":
    main()
