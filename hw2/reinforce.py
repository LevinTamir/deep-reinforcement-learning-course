import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TrainConfig:
    env_name: str = "CartPole-v1"
    seed: int = 123
    gamma: float = 0.99

    lr_actor: float = 2e-3
    lr_critic: float = 2e-3

    hidden: int = 256
    max_episodes: int = 2000

    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float | None = 1.0

    use_baseline: bool = False

    device: str = "cpu"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    Monte-Carlo discounted returns.
    Given rewards r_0..r_{T-1}, returns a tensor G_0..G_{T-1}.
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


class PolicyNet(nn.Module):
    """
    Stochastic policy pi_theta(a|s) for *discrete* action spaces.
    Outputs logits for a categorical distribution.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    """State-value function approximator V_phi(s)."""
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch]


@torch.no_grad()
def evaluate(policy: PolicyNet, env, episodes: int, device: str) -> float:
    """
    Greedy-ish evaluation: take argmax over action probabilities.
    (Alternatively sample for stochastic eval; choose what course expects.)
    """
    policy.eval()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(x)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    policy.train()
    return float(np.mean(returns))


def run_episode(policy: PolicyNet, env, device: str) -> dict[str, list]:
    """
    Collect one full trajectory:
    stores states, log_probs, rewards, and entropy terms.
    """
    obs, _ = env.reset()
    done = False

    states, log_probs, rewards, entropies = [], [], [], []

    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        logits = policy(x)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(x)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        entropies.append(entropy)

        obs = next_obs

    return {
        "states": states,
        "log_probs": log_probs,
        "rewards": rewards,
        "entropies": entropies,
    }


def moving_average(x: list[float], window: int) -> np.ndarray:
    """Simple moving average using convolution; returns an array shorter by window-1."""
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
    ma_window: int ,
    title: str
) -> None:
    """
    Plots:
      - Reward per episode
      - Average reward over last 100 episodes
      - (Optional) moving average of reward per episode
      - Eval average reward every eval_every episodes
    This matches the 'reward per episode' and 'average reward in the last 100 episodes'
    style used in the reference report. [file:144][web:82]
    """
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

def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    if cfg.use_baseline:
        normalize_returns = False
        normalize_advantages = True
    else:
        normalize_returns = True
        normalize_advantages = False

    env = gym.make(cfg.env_name)
    eval_env = gym.make(cfg.env_name)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)

    device = torch.device(cfg.device)

    policy = PolicyNet(obs_dim, act_dim, cfg.hidden).to(device)
    opt_actor = optim.Adam(policy.parameters(), lr=cfg.lr_actor)

    value = None
    opt_critic = None
    if cfg.use_baseline:
        value = ValueNet(obs_dim, cfg.hidden).to(device)
        opt_critic = optim.Adam(value.parameters(), lr=cfg.lr_critic)

    train_returns = []
    avg100_returns = []
    eval_returns = []
    eval_every = 25
    solve_ep = None

    for ep in range(cfg.max_episodes):
        traj = run_episode(policy, env, cfg.device)

        train_return = float(sum(traj["rewards"]))
        train_returns.append(train_return)

        if len(train_returns) >= 100:
            avg100 = float(np.mean(train_returns[-100:]))
        else:
            avg100 = float("nan")
        avg100_returns.append(avg100)

        if solve_ep is None and len(train_returns) >= 100 and avg100 > 475.0:
            solve_ep = ep + 1

            print("SOLVED at episode", ep + 1)
            break

        log_probs = torch.stack(traj["log_probs"]).to(device)
        entropies = torch.stack(traj["entropies"]).to(device)
        states = torch.stack(traj["states"]).to(device)

        returns_raw = compute_returns(traj["rewards"], cfg.gamma).to(device)

        if not cfg.use_baseline:
            if normalize_returns:
                returns_for_actor = (returns_raw - returns_raw.mean()) / (returns_raw.std() + 1e-8)
            else:
                returns_for_actor = returns_raw

            policy_loss = -(log_probs * returns_for_actor).sum()
            entropy_bonus = entropies.sum()
            loss = policy_loss - cfg.entropy_coef * entropy_bonus

            opt_actor.zero_grad()
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt_actor.step()

        else:
            values = value(states)
            advantages = returns_raw - values

            if normalize_advantages:
                advantages_for_actor = (advantages.detach() - advantages.detach().mean()) / (advantages.detach().std() + 1e-8)
            else:
                advantages_for_actor = advantages.detach()

            policy_loss = -(log_probs * advantages_for_actor).sum()
            value_loss = 0.5 * ((returns_raw.detach() - values) ** 2).sum()
            entropy_bonus = entropies.sum()

            total_loss = (policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy_bonus)

            opt_actor.zero_grad()
            opt_critic.zero_grad()
            total_loss.backward()

            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(value.parameters(), cfg.max_grad_norm)

            opt_actor.step()
            opt_critic.step()

        if (ep + 1) % eval_every == 0:
            avg_eval = evaluate(policy, eval_env, episodes=5, device=cfg.device)
            eval_returns.append(avg_eval)
            print(
                f"Episode {ep+1:5d} | "
                f"train_return={train_return:8.2f} | "
                f"avg100={avg100:8.2f} | "
                f"eval_avg={avg_eval:8.2f}"
            )

    tag = "with_baseline" if cfg.use_baseline else "without_baseline"

    plot_learning_curves(
        train_returns=train_returns,
        avg100_returns=avg100_returns,
        eval_returns=eval_returns,
        eval_every=eval_every,
        out_path=f"plots/reinforce_{tag}.png",
        ma_window=50,
        title=f"REINFORCE ({tag}) - {cfg.env_name}",
    )

    # Save results for comparison
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / f"reinforce_{tag}.npz",
        train_returns=np.array(train_returns),
        avg100_returns=np.array(avg100_returns),
        eval_returns=np.array(eval_returns),
        eval_every=eval_every,
        solved_ep=solve_ep if solve_ep is not None else -1,
    )
    print(f"Results saved to results/reinforce_{tag}.npz")

    env.close()
    eval_env.close()

    if solve_ep is not None:
        print(f"SOLVED ({tag}) at episode {solve_ep} with avg100 > 475.")
    else:
        print(f"NOT SOLVED ({tag}) within {cfg.max_episodes} episodes (avg100 never exceeded 475).")

    return train_returns, avg100_returns, eval_returns, solve_ep



if __name__ == "__main__":
    print("Running REINFORCE without baseline")
    cfg = TrainConfig(
        env_name="CartPole-v1",
        use_baseline=False,
        lr_actor=3e-4,
        entropy_coef=0.0,
        device="cpu",
    )
    train(cfg)

    print("*******************************")

    print("Running REINFORCE with baseline")
    cfg = TrainConfig(
        env_name="CartPole-v1",
        use_baseline=True,
        lr_actor=5e-4,
        lr_critic=1e-3,
        entropy_coef=0.01,
        device="cpu",
    )
    train(cfg)

