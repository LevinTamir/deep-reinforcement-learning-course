# ============================================
# Section 3 – Progressive Transfer {Acrobot, MountainCar}
# Mark Feldman (320827637) & Tamir Levin (315765347)
# ============================================
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
class ProgressiveConfig:
    source1_name: str = "Acrobot"
    source2_name: str = "MountainCar"
    source1_path: str = "models/acrobot_actor.pt"
    source2_path: str = "models/mountaincar_actor.pt"
    
    target_env: str = "CartPole-v1"
    
    fixed_obs_dim: int = 6
    fixed_act_dim: int = 3
    hidden: int = 256
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_actor: float = 2e-3
    lr_critic: float = 5e-3
    lr_step_size: int = 75
    lr_gamma: float = 0.9
    min_lr: float = 1e-4
    
    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    
    max_episodes: int = 2000
    seed: int = 42
    
    print_every: int = 10
    eval_every: int = 25
    eval_episodes: int = 10
    solve_score: float = 475.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def adapter_init(layer, std=0.1):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


def pad_observation(obs: np.ndarray, target_size: int = 6) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if len(obs) < target_size:
        obs = np.pad(obs, (0, target_size - len(obs)), mode='constant', constant_values=0.0)
    return obs[:target_size]


class SourceActor(nn.Module):
    """
    frozen source actor network 
    loads pretrained weights and exposes ALL hidden layer features for lateral connections
    """
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x)
    
    def get_all_hidden_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #extract features from both hidden layers for lateral connections
        h1 = torch.relu(self.fc1(x))
        h2 = torch.nn.functional.elu(self.fc2(h1))
        return h1, h2
    
    def get_hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        #extract features from the top hidden layer (fc2 output)
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return x
    
    def load_pretrained(self, path: str) -> None:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class ProgressiveActor(nn.Module):

    """
    progressive networks actor with gated lateral connections.
    
    architecture:
    two frozen source networks (Acrobot, MountainCar)
    one trainable target network
    learnable gates that control how much source knowledge flows in
    gates initialized near zero to allow target to learn first, then gradually use sources
    """
    def __init__(self, cfg: ProgressiveConfig):
        super().__init__()
        
        self.source1 = SourceActor(cfg.fixed_obs_dim, cfg.fixed_act_dim, cfg.hidden)
        self.source2 = SourceActor(cfg.fixed_obs_dim, cfg.fixed_act_dim, cfg.hidden)
        
        self.fc1 = layer_init(nn.Linear(cfg.fixed_obs_dim, cfg.hidden))
        self.fc2 = layer_init(nn.Linear(cfg.hidden, cfg.hidden))
        
        self.adapter1 = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh()
        )
        self.adapter2 = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh()
        )
        
        for adapter in [self.adapter1, self.adapter2]:
            for m in adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.0)
        

        self.gate1 = nn.Parameter(torch.tensor(-2.0))
        self.gate2 = nn.Parameter(torch.tensor(-2.0))

        self.fc3 = layer_init(nn.Linear(cfg.hidden, cfg.fixed_act_dim), std=0.01)
        
        self.hidden_size = cfg.hidden
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            source1_h = self.source1.get_hidden_features(x)
            source2_h = self.source2.get_hidden_features(x)
        
        target_h = torch.relu(self.fc1(x))
        target_h = torch.nn.functional.elu(self.fc2(target_h))
        
        gate1 = torch.sigmoid(self.gate1)
        gate2 = torch.sigmoid(self.gate2)
        
        adapted1 = gate1 * self.adapter1(source1_h)
        adapted2 = gate2 * self.adapter2(source2_h)
        
        combined = target_h + adapted1 + adapted2
        
        return self.fc3(combined)
    
    def load_source_networks(self, source1_path: str, source2_path: str) -> None:
        self.source1.load_pretrained(source1_path)
        self.source2.load_pretrained(source2_path)


class Critic(nn.Module):
    #value function V(s; w) - standard architecture
    def __init__(self, obs_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def step_env(env, api: str, action: int):
    obs2, reward, terminated, truncated, info = env.step(action)
    return obs2, float(reward), bool(terminated), bool(truncated), info


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
    gae_lambda: float
) -> torch.Tensor:

    """
    compute GAE advantages
    A_t^GAE = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
    where δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    
    advantages = []
    gae = 0.0
    
    for i in reversed(range(len(rewards))):
        r = rewards[i]
        v = values[i].item()
        v_next = next_values[i].item() if not dones[i] else 0.0
        
        delta = r + gamma * v_next - v

        if dones[i]:
            gae = delta
        else:
            gae = delta + gamma * gae_lambda * gae
        
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)


@torch.no_grad()
def evaluate_policy(cfg: ProgressiveConfig, actor: ProgressiveActor, episodes: int = 10) -> float:
    was_training = actor.training
    actor.eval()
    env = gym.make(cfg.target_env)
    
    act_dim = int(env.action_space.n)

    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + 10000 + i)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(pad_observation(obs), dtype=torch.float32)
            logits = actor(obs_t)
            action = int(torch.argmax(logits).item()) % act_dim
            obs, r, terminated, truncated, _ = step_env(env, "gymnasium", action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    if was_training:
        actor.train()
    return float(np.mean(returns))


def train_progressive(cfg: ProgressiveConfig) -> tuple[list[float], int, float, float, int]:

    start_time = time.time()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    env = gym.make(cfg.target_env)
    obs, _ = env.reset(seed=cfg.seed)

    obs_dim = int(np.asarray(obs).shape[0])
    act_dim = int(env.action_space.n)


    actor = ProgressiveActor(cfg)
    actor.load_source_networks(cfg.source1_path, cfg.source2_path)
    
    trainable_params = [p for p in actor.parameters() if p.requires_grad]

    critic = Critic(cfg.fixed_obs_dim, cfg.hidden)

    opt_actor = torch.optim.Adam(trainable_params, lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

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
            
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item()) % act_dim
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()

            value = critic(obs_t)

            obs2, reward, terminated, truncated, _ = step_env(env, "gymnasium", action)
            done = terminated or truncated

            with torch.no_grad():
                obs2_t = torch.as_tensor(pad_observation(obs2), dtype=torch.float32)
                next_value = critic(obs2_t)

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
        total_iterations += len(rewards)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        values_t = torch.stack(values)

        advantages = compute_gae_advantages(rewards, values, next_values, dones, cfg.gamma, cfg.gae_lambda)

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
        eval_returns=eval_returns,
        eval_every=cfg.eval_every,
        out_path="plots/q3a_progressive_to_cartpole.png",
        ma_window=50,
        title=f"Progressive Networks ({cfg.source1_name}, {cfg.source2_name}) → CartPole",
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / "progressive_to_cartpole.npz",
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
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), models_dir / "progressive_to_cartpole_actor.pt")

    return episode_returns, solved_ep, best_greedy, elapsed_time, total_iterations


def main():
    cfg = ProgressiveConfig(
        source1_name="Acrobot",
        source2_name="MountainCar",
        source1_path="models/acrobot_actor.pt",
        source2_path="models/mountaincar_actor.pt",
        
        target_env="CartPole-v1",
        
        fixed_obs_dim=6,
        fixed_act_dim=3,
        hidden=256,
        
        gamma=0.99,
        gae_lambda=0.95,
        lr_actor=2e-3,
        lr_critic=5e-3,
        lr_step_size=75,
        lr_gamma=0.9,
        min_lr=1e-4,
        
        entropy_coef=0.02,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=True,
        
        max_episodes=2000,
        seed=42,
        print_every=10,
        eval_every=25,
        eval_episodes=10,
        solve_score=475.0,
    )

    
    train_progressive(cfg)



if __name__ == "__main__":
    main()
