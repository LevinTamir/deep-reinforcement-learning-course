
from dataclasses import dataclass
from copy import deepcopy
import time

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym

from plotting_utils import *
@dataclass
class TransferConfig:
    source_env: str = "Acrobot-v1"
    target_env: str = "CartPole-v1"
    
    pretrained_path: str = "models/acrobot_actor.pt"
    
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    hidden: int = 256
    
    lr_step_size: int = 100
    lr_gamma: float = 0.95
    min_lr: float = 1e-5
    
    entropy_coef: float = 0.05
    value_loss_coef: float = 0.25
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    
    use_gae: bool = False
    gae_lambda: float = 0.95
    
    use_reward_shaping: bool = False
    
    freeze_hidden_layers: bool = True
    reinit_output_layer: bool = True
    
    max_episodes: int = 1000
    seed: int = 123
    
    print_every: int = 10
    eval_every: int = 25
    eval_episodes: int = 10
    solve_score: float = 475.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def pad_observation(obs: np.ndarray, target_size: int = 6) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if len(obs) < target_size:
        obs = np.pad(obs, (0, target_size - len(obs)), mode='constant', constant_values=0.0)
    return obs[:target_size]


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 256):
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
    def __init__(self, obs_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.obs_dim = 6
        self.fc1 = layer_init(nn.Linear(self.obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def step_env(env, api: str, action: int):
    obs2, reward, terminated, truncated, info = env.step(action)
    return obs2, float(reward), bool(terminated), bool(truncated), info


def moving_average(x: list[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

def compute_td_advantages(
    rewards: list[float],
    values: list[torch.Tensor],
    next_values: list[torch.Tensor],
    dones: list[bool],
    gamma: float
) -> torch.Tensor:

    advantages = []
    for r, v, v_next, done in zip(rewards, values, next_values, dones):
        bootstrap = 0.0 if done else v_next.item()
        td_target = r + gamma * bootstrap
        td_error = td_target - v.item()
        advantages.append(td_error)
    return torch.tensor(advantages, dtype=torch.float32)


def compute_gae_advantages(
    rewards: list[float],
    values: list[torch.Tensor],
    next_values: list[torch.Tensor],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        v = float(values[t].item())
        v_next = 0.0 if dones[t] else float(next_values[t].item())
        delta = float(rewards[t]) + gamma * v_next - v
        gae = delta + gamma * gae_lambda * (1 - int(dones[t])) * gae
        advantages[t] = float(gae)
    return advantages


def shaped_reward_cartpole(obs: np.ndarray, obs2: np.ndarray, r_env: float, done: bool) -> float:

    pole_angle = obs2[2]
    
    angle_bonus = 0.1 * (1.0 - abs(pole_angle) / 0.2095)
    
    termination_penalty = -5.0 if done else 0.0
    
    return r_env + angle_bonus + termination_penalty


@torch.no_grad()
def evaluate_policy(cfg: TransferConfig, actor: Actor, episodes: int = 10) -> float:
    was_training = actor.training
    actor.eval()
    env = gym.make(cfg.target_env)

    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + 10000 + i)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(pad_observation(obs), dtype=torch.float32)
            logits = actor(obs_t)
            action = int(torch.argmax(logits[:2]).item())
            obs, r, terminated, truncated, _ = step_env(env, "gymnasium", action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    if was_training:
        actor.train()
    return float(np.mean(returns))


def train_transfer(cfg: TransferConfig) -> tuple[list[float], int, float, float, int]:
    start_time = time.time()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    actor = Actor(6, 3, cfg.hidden)
    critic = Critic(6, cfg.hidden)
    
    pretrained_path = Path(cfg.pretrained_path)
    if pretrained_path.exists():
        actor.load_state_dict(torch.load(pretrained_path, weights_only=True))
        
        if cfg.reinit_output_layer:
            actor.fc3 = layer_init(nn.Linear(cfg.hidden, 3), std=0.01)
        
        if cfg.freeze_hidden_layers:
            for param in actor.fc1.parameters():
                param.requires_grad = False
            for param in actor.fc2.parameters():
                param.requires_grad = False

    
    pretrained_eval = evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)

    env = gym.make(cfg.target_env)
    obs, _ = env.reset(seed=cfg.seed)
    act_dim = int(env.action_space.n)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    
    scheduler_actor = StepLR(opt_actor, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    scheduler_critic = StepLR(opt_critic, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    episode_returns = []
    avg100_returns = []
    eval_returns = [pretrained_eval]
    solved_ep = -1
    best_greedy = pretrained_eval
    best_actor_state = deepcopy(actor.state_dict())
    total_iterations = 0

    for ep in range(cfg.max_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        
        log_probs = []
        entropies = []
        values = []
        next_values = []
        rewards = []
        dones = []

        while not done:
            obs_t = torch.as_tensor(pad_observation(obs), dtype=torch.float32)
            
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item()) % act_dim
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()
            
            value = critic(obs_t)
            
            obs2, reward, terminated, truncated, _ = step_env(env, "gymnasium", action)
            done = terminated or truncated
            
            if cfg.use_reward_shaping:
                reward = shaped_reward_cartpole(obs, obs2, reward, done)
            
            with torch.no_grad():
                obs2_t = torch.as_tensor(pad_observation(obs2), dtype=torch.float32)
                next_value = critic(obs2_t)
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            dones.append(terminated)
            
            obs = obs2

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)
        total_iterations += len(rewards)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        values_t = torch.stack(values)

        if cfg.use_gae:
            advantages = compute_gae_advantages(rewards, values, next_values, dones, cfg.gamma, cfg.gae_lambda)
        else:
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
    
    evaluate_policy(cfg, actor, episodes=cfg.eval_episodes)

    plot_learning_curves(
        train_returns=episode_returns,
        avg100_returns=avg100_returns,
        eval_returns=eval_returns[1:],
        eval_every=cfg.eval_every,
        out_path="plots/q2a_transfer_acrobot_to_cartpole.png",
        title=f"Transfer Learning: {cfg.source_env} → {cfg.target_env}",
    )
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "transfer_acrobot_to_cartpole.npz",
        train_returns=np.array(episode_returns),
        avg100_returns=np.array(avg100_returns),
        eval_returns=np.array(eval_returns),
        eval_every=cfg.eval_every,
        pretrained_eval=pretrained_eval,
        solved_ep=solved_ep if solved_ep != -1 else -1,
        best_greedy=best_greedy,
        elapsed_time=elapsed_time,
        total_iterations=total_iterations,
    )
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), models_dir / "transfer_acrobot_to_cartpole_actor.pt")
    
    return episode_returns, solved_ep, best_greedy, elapsed_time, total_iterations


def main():
    cfg = TransferConfig(
        source_env="Acrobot-v1",
        target_env="CartPole-v1",
        pretrained_path="models/acrobot_actor.pt",
        
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        hidden=256,
        
        lr_step_size=100,
        lr_gamma=0.95,
        min_lr=1e-5,
        
        entropy_coef=0.05,
        value_loss_coef=0.25,
        max_grad_norm=0.5,
        normalize_advantages=True,
        
        use_gae=True,
        gae_lambda=0.95,
        use_reward_shaping=True,
        
        freeze_hidden_layers=False,
        reinit_output_layer=False,
        
        max_episodes=1000,
        seed=42,
        eval_every=25,
        eval_episodes=10,
        solve_score=475.0,
    )

    
    train_transfer(cfg)


if __name__ == "__main__":
    main()
