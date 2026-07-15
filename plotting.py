import os

import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([])

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(
    env_name: str,
    episode_rewards: list[float],
    update_history: dict[str, list[float]],
    save_dir: str = "figures",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    rolling = _rolling_mean(episode_rewards, window=20)
    if rolling.size:
        ax.plot(range(19, len(episode_rewards)), rolling, label="Rolling mean (20)")
    ax.set_title("Episode reward")
    ax.set_xlabel("Episode")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(update_history["average_reward"])
    ax.set_title("Average reward (last 100 episodes)")
    ax.set_xlabel("Update")

    ax = axes[0, 2]
    ax.plot(update_history["actor_loss"])
    ax.set_title("Actor loss")
    ax.set_xlabel("Update")

    ax = axes[1, 0]
    ax.plot(update_history["critic_loss"])
    ax.set_title("Critic loss")
    ax.set_xlabel("Update")

    ax = axes[1, 1]
    ax.plot(update_history["entropy"])
    ax.set_title("Policy entropy")
    ax.set_xlabel("Update")

    ax = axes[1, 2]
    ax.plot(update_history["clip_fraction"], label="Clip fraction")
    ax.plot(update_history["ratio_mean"], label="Ratio mean")
    ax.set_title("PPO clip fraction / ratio mean")
    ax.set_xlabel("Update")
    ax.legend()

    fig.suptitle(env_name)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"training_curves_{env_name}.png"), dpi=150)
    plt.close(fig)
