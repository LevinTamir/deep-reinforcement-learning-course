# ============================================================
#  Section 3b – Progressive Networks Transfer Learning
#  Task: {CartPole, Acrobot} → MountainCarContinuous (continuous target)
# ============================================================

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
from sklearn.preprocessing import StandardScaler


@dataclass
class ProgressiveConfig:
    # Source models (frozen)
    source1_name: str = "CartPole"
    source2_name: str = "Acrobot"
    source1_path: str = "models/cartpole_actor.pt"
    source2_path: str = "models/acrobot_actor.pt"
    
    # Target environment
    target_env: str = "MountainCarContinuous-v0"
    
    # Architecture (must match source networks)
    fixed_obs_dim: int = 6
    fixed_act_dim: int = 3
    hidden: int = 256
    
    # Training hyperparameters
    gamma: float = 0.985
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    lr_step_size: int = 100
    lr_gamma: float = 0.95
    min_lr: float = 1e-5
    
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = False
    
    max_episodes: int = 2000
    seed: int = 123
    
    print_every: int = 10
    eval_every: int = 10
    eval_episodes: int = 10
    solve_score: float = 90.0
    
    # Gaussian policy settings
    log_std_min: float = -3.0
    log_std_max: float = 1.0
    
    # Optional: per-step discount accumulator I_t
    use_I_weight: bool = True
    
    # Reward shaping toggle (learning only)
    use_reward_shaping: bool = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_observation(obs: np.ndarray, target_size: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[0] < target_size:
        obs = np.pad(obs, (0, target_size - obs.shape[0]), mode="constant", constant_values=0.0)
    return obs[:target_size]


def step_env(env, action: np.ndarray):
    """Step environment and return standardized output."""
    obs2, reward, terminated, truncated, info = env.step(action)
    return obs2, float(reward), bool(terminated), bool(truncated), info


def moving_average(x: list[float], window: int) -> np.ndarray:
    """Simple moving average."""
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
    out_path: str,
    ma_window: int = 50,
    title: str = "Progressive Networks Learning Curve",
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


class SourceActor(nn.Module):
    """
    Frozen source actor network. 
    Loads pretrained weights and exposes hidden layer features for lateral connections.
    """
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x)
    
    def get_hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the top hidden layer (fc2 output)."""
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return x
    
    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights and freeze all parameters."""
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class ProgressiveActor(nn.Module):
    """
    Progressive Networks Actor with lateral connections from frozen source networks.
    For continuous actions (MountainCarContinuous).
    
    Architecture:
    - Two frozen source networks (CartPole, Acrobot)
    - One trainable target network
    - Lateral connections from source hidden layers to target output
    
    Output: (mu, log_std, dummy) for Gaussian policy
    """
    def __init__(self, cfg: ProgressiveConfig):
        super().__init__()
        
        # Frozen source networks
        self.source1 = SourceActor(cfg.fixed_obs_dim, cfg.fixed_act_dim, cfg.hidden)
        self.source2 = SourceActor(cfg.fixed_obs_dim, cfg.fixed_act_dim, cfg.hidden)
        
        # Trainable target network layers
        self.fc1 = layer_init(nn.Linear(cfg.fixed_obs_dim, cfg.hidden))
        self.fc2 = layer_init(nn.Linear(cfg.hidden, cfg.hidden))
        
        # Output layer: receives target hidden + lateral connections from both sources
        # Input: hidden (target fc2) + hidden (source1 fc2) + hidden (source2 fc2)
        self.fc3 = layer_init(nn.Linear(cfg.hidden * 3, cfg.fixed_act_dim), std=0.01)
        
        # Mild bias to avoid "do nothing" optimum
        with torch.no_grad():
            self.fc3.bias[0].fill_(0.2)   # mu bias (small push)
            self.fc3.bias[1].fill_(-0.5)  # log_std bias (moderate std)
        
        self.hidden_size = cfg.hidden
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get frozen features from source networks
        with torch.no_grad():
            source1_h = self.source1.get_hidden_features(x)
            source2_h = self.source2.get_hidden_features(x)
        
        # Target network hidden layers (trainable)
        target_h = torch.relu(self.fc1(x))
        target_h = torch.nn.functional.elu(self.fc2(target_h))
        
        # Combine target hidden with lateral connections from sources
        combined = torch.cat([target_h, source1_h, source2_h], dim=-1)
        
        # Final output
        return self.fc3(combined)
    
    def load_source_networks(self, source1_path: str, source2_path: str) -> None:
        """Load pretrained weights for both source networks."""
        print(f"Loading source 1 from: {source1_path}")
        self.source1.load_pretrained(source1_path)
        print(f"Loading source 2 from: {source2_path}")
        self.source2.load_pretrained(source2_path)
        print("Source networks loaded and frozen.")


class Critic(nn.Module):
    """Value function V(s; w)."""
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def sample_action_and_logprob(cfg: ProgressiveConfig, actor_out: torch.Tensor):
    """
    Stochastic policy for continuous actions:
      - Sample a ~ Normal(mu, std)
      - Use tanh(a) to bound into [-1, 1]
    """
    mu = actor_out[0]
    log_std = actor_out[1].clamp(cfg.log_std_min, cfg.log_std_max)
    std = torch.exp(log_std)

    dist = torch.distributions.Normal(mu, std)
    pre_tanh = dist.rsample()
    action = torch.tanh(pre_tanh)

    # log pi(a) with tanh correction
    log_prob_u = dist.log_prob(pre_tanh)
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = (log_prob_u - correction).squeeze()
    entropy = dist.entropy().squeeze()

    return action.unsqueeze(0), log_prob, entropy


def compute_td_advantages(
    rewards: list[float],
    values: list[torch.Tensor],
    next_values: list[torch.Tensor],
    dones: list[bool],
    gamma: float,
) -> torch.Tensor:
    """Compute TD(0) advantages: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)"""
    advantages = []
    for r, v, v_next, done in zip(rewards, values, next_values, dones):
        bootstrap = 0.0 if done else v_next.item()
        td_target = r + gamma * bootstrap
        td_error = td_target - v.item()
        advantages.append(td_error)
    return torch.tensor(advantages, dtype=torch.float32)


def shaped_reward_energy(obs: np.ndarray, obs2: np.ndarray, r_env: float) -> float:
    """
    Energy-based reward shaping for MountainCar (learning only).
    
    MATCHES Q1 MOUNTAINCAR: Uses the same energy bonus coefficients and scaling
    to ensure consistent learning dynamics for transfer learning alignment.
    """
    height_old = np.sin(3 * obs[0])
    height_new = np.sin(3 * obs2[0])

    kinetic_old = 0.5 * obs[1] ** 2
    kinetic_new = 0.5 * obs2[1] ** 2

    # Q1 MountainCar coefficient: 100.0
    energy_bonus = 100.0 * ((height_new + 50 * kinetic_new) - (height_old + 50 * kinetic_old))

    if obs2[0] >= 0.45:
        energy_bonus += 100.0
    elif obs2[0] > 0.3:
        energy_bonus += 10.0

    return float(r_env) + float(energy_bonus)


@torch.no_grad()
def evaluate_policy_greedy(
    cfg: ProgressiveConfig, 
    actor: ProgressiveActor, 
    scaler: StandardScaler, 
    episodes: int
) -> float:
    """Greedy evaluation on TRUE env reward: a = tanh(mu)"""
    env = gym.make(cfg.target_env)
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
            action = torch.tanh(mu)
            action_np = action.cpu().numpy().astype(np.float32).reshape(1,)

            obs, r_env, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            ep_ret_env += float(r_env)

        returns.append(ep_ret_env)

    env.close()
    if actor_was_training:
        actor.train()
    return float(np.mean(returns))


def train_progressive(cfg: ProgressiveConfig) -> tuple[list[float], int, float, float, int]:
    """
    Train Progressive Networks actor on MountainCarContinuous.
    
    Returns:
        episode_returns: list of returns per episode
        solved_ep: episode at which solve threshold was reached (-1 if not solved)
        best_greedy: best greedy evaluation return
        elapsed_time: total training time in seconds
        num_iterations: total number of training iterations
    """
    start_time = time.time()
    set_seed(cfg.seed)

    # Check if source models exist
    source1_exists = Path(cfg.source1_path).exists()
    source2_exists = Path(cfg.source2_path).exists()
    
    if not source1_exists:
        print(f"ERROR: Source model not found: {cfg.source1_path}")
        print(f"Please run q1_cartpole_actor_critic.py first to train the CartPole model.")
        return [], -1, 0.0, 0.0, 0
    
    if not source2_exists:
        print(f"ERROR: Source model not found: {cfg.source2_path}")
        print(f"Please run q1_acrobot_actor_critic.py first to train the Acrobot model.")
        return [], -1, 0.0, 0.0, 0

    env = gym.make(cfg.target_env)
    obs0, _ = env.reset(seed=cfg.seed)

    # Fit scaler on actual trajectories
    print("Collecting initial trajectories for scaler...")
    scaler_obs = []
    for _ in range(50):
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

    obs_dim = int(np.asarray(obs0).shape[0])

    print(f"\n=== Progressive Networks Transfer Learning ===")
    print(f"Source 1: {cfg.source1_name} ({cfg.source1_path})")
    print(f"Source 2: {cfg.source2_name} ({cfg.source2_path})")
    print(f"Target: {cfg.target_env}")
    print(f"Original observation size: {obs_dim}")
    print(f"Padded observation size (input): {cfg.fixed_obs_dim}")
    print(f"Network output size (fixed): {cfg.fixed_act_dim}")
    print("Action bounds: [-1.0, 1.0]")
    print(f"Hidden layer size: {cfg.hidden}")

    # Create Progressive Actor and load source networks
    actor = ProgressiveActor(cfg)
    actor.load_source_networks(cfg.source1_path, cfg.source2_path)
    
    # Only target network parameters are trained (source networks are frozen)
    trainable_params = [p for p in actor.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    critic = Critic(cfg.fixed_obs_dim, cfg.hidden)

    # Adam optimizer (only for trainable parameters)
    opt_actor = torch.optim.Adam(trainable_params, lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # Learning rate schedulers
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
        rewards = []       # Learning rewards (shaped if enabled)
        rewards_env = []   # TRUE env rewards (for reporting)
        dones = []
        I_weights = []
        I = 1.0

        while not done:
            # Normalize + pad
            obs_norm = scaler.transform(np.asarray(obs, dtype=np.float32).reshape(1, -1)).flatten()
            obs_pad = pad_observation(obs_norm, cfg.fixed_obs_dim)
            obs_t = torch.as_tensor(obs_pad, dtype=torch.float32)

            # Actor: sample action
            out = actor(obs_t)
            action_t, log_prob, entropy = sample_action_and_logprob(cfg, out)

            # Critic: estimate value
            value = critic(obs_t)

            # Environment step
            action_np = action_t.detach().cpu().numpy().astype(np.float32)
            obs2, r_env, terminated, truncated, _ = step_env(env, action_np)
            done = terminated or truncated

            # Learning reward (optionally shaped)
            if cfg.use_reward_shaping:
                r_learn = shaped_reward_energy(obs, obs2, r_env)
            else:
                r_learn = r_env

            # Next state value
            with torch.no_grad():
                obs2_norm = scaler.transform(np.asarray(obs2, dtype=np.float32).reshape(1, -1)).flatten()
                obs2_pad = pad_observation(obs2_norm, cfg.fixed_obs_dim)
                obs2_t = torch.as_tensor(obs2_pad, dtype=torch.float32)
                next_value = critic(obs2_t)

            # Store transition
            states.append(obs_t)
            actions.append(action_t)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            next_values.append(next_value)
            rewards.append(r_learn)
            rewards_env.append(r_env)
            dones.append(terminated)
            I_weights.append(I)

            obs = obs2
            I *= cfg.gamma

        ep_ret = sum(rewards_env)  # TRUE env return
        episode_returns.append(ep_ret)
        total_iterations += len(rewards)

        # Convert to tensors
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        values_t = torch.stack(values)

        # Compute advantages using learning rewards
        advantages = compute_td_advantages(rewards, values, next_values, dones, cfg.gamma)

        if cfg.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if cfg.use_I_weight:
            I_t = torch.tensor(I_weights, dtype=torch.float32)
        else:
            I_t = torch.ones_like(advantages)

        # Actor loss
        actor_loss = -(log_probs_t * advantages * I_t).mean()
        entropy_loss = -cfg.entropy_coef * entropies_t.mean()

        # Critic loss (TD targets using learning rewards)
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
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
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

        # Greedy evaluation on TRUE env return
        if (ep + 1) % cfg.eval_every == 0:
            last_greedy = evaluate_policy_greedy(cfg, actor, scaler, cfg.eval_episodes)
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

    # Restore best actor
    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)

    final_eval = evaluate_policy_greedy(cfg, actor, scaler, cfg.eval_episodes)
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
        out_path="plots/progressive_to_mountaincar.png",
        ma_window=50,
        title=f"Progressive Networks ({cfg.source1_name}, {cfg.source2_name}) → MountainCar",
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "progressive_to_mountaincar.npz",
        train_returns=np.array(episode_returns),
        avg100_returns=np.array(avg100_returns),
        eval_returns=np.array(eval_returns),
        eval_every=cfg.eval_every,
        solved_ep=solved_ep if solved_ep != -1 else -1,
        best_greedy=best_greedy,
        elapsed_time=elapsed_time,
        total_iterations=total_iterations,
        source1=cfg.source1_name,
        source2=cfg.source2_name,
        used_reward_shaping=cfg.use_reward_shaping,
    )

    # Save trained model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), models_dir / "progressive_to_mountaincar_actor.pt")
    print(f"Saved model to {models_dir}/progressive_to_mountaincar_actor.pt")

    return episode_returns, solved_ep, best_greedy, elapsed_time, total_iterations


def main():
    cfg = ProgressiveConfig(
        # Source networks
        source1_name="CartPole",
        source2_name="Acrobot",
        source1_path="models/cartpole_actor.pt",
        source2_path="models/acrobot_actor.pt",
        
        # Target
        target_env="MountainCarContinuous-v0",
        
        # Architecture
        fixed_obs_dim=6,
        fixed_act_dim=3,
        hidden=256,
        
        # Training
        gamma=0.985,
        lr_actor=2e-5,      # Much smaller (reference: 0.00002)
        lr_critic=1e-3,     # Larger (reference: 0.001) - bigger critic:actor ratio
        lr_step_size=60,    # Decay every 60 episodes (reference pattern)
        lr_gamma=0.7,       # Multiply by 0.7 each time (reference)
        min_lr=1e-6,
        
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=False,
        
        max_episodes=2000,
        seed=123,
        print_every=10,
        eval_every=10,
        eval_episodes=10,
        solve_score=90.0,
        
        log_std_min=-3.0,
        log_std_max=1.0,
        use_I_weight=True,
        use_reward_shaping=True,
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
    
    returns, solved_ep, best_greedy, elapsed_time, total_iterations = train_progressive(cfg)
    
    if returns:
        print(f"\nDone. episodes={len(returns)}, solved_ep={solved_ep}, best_greedy={best_greedy:.1f}")
        print(f"Elapsed time: {elapsed_time:.2f}s, Total iterations: {total_iterations}")


if __name__ == "__main__":
    main()
