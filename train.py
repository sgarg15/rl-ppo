import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import deque

from ac import ActorCriticAgent

ENV_NAME = "CartPole-v1"
SEED = 42

GAMMA = 0.99 # Discount factor for future rewards
LEARNING_RATE = 0.01 # Learning rate for the optimizer

VALUE_COEF = 0.5 # Coefficient for the value loss
ENTROPY_COEF = 0.001 # Coefficient for the entropy loss

POSITION_PENALTY_COEF = 0.1 # Coefficient for penalizing distance of the cart from the center

MAX_EPISODES = 1000 # Maximum number of episodes to train
HIDDEN_DIM = 128 # Number of hidden units in the neural network

ROLLOUT_STEPS = 256 # Number of steps to rollout before updating the model

MAX_GRAD_NORM = 0.5 # Maximum gradient norm for gradient clipping

GAE_LAMBDA = 0.95 # Lambda parameter for Generalized Advantage Estimation (GAE)

CLIP_EPSILON = 0.2 # Clipping parameter for PPO

PPO_EPOCHS = 4 # Number of epochs to update the model per rollout

def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([])

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(
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

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"training_curves_{ENV_NAME}.png"), dpi=150)
    plt.close(fig)


def train() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticAgent(observation_dim, action_dim, HIDDEN_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    state, _ = env.reset(seed=SEED)
    current_episode_reward = 0.0
    reward_history = deque(maxlen=100)
    episode_completed = 0

    episode_rewards = []
    update_history = {
        "average_reward": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "ratio_mean": [],
    }

    for update in range(MAX_EPISODES):
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_states = []
        rollout_terminated = []
        rollout_episode_ended = []
        rollout_old_log_probs = []

        for step in range(ROLLOUT_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                action_logits, state_value = model(state_tensor)

                # Create a categorical distribution over the action logits
                distribution = Categorical(logits=action_logits)

                # Sample an action from the distribution and compute its log probability and compute the entropy of the distribution
                action = distribution.sample()
                
                # Compute the log probability of the selected action
                old_log_prob = distribution.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            current_episode_reward += reward

            cart_position = next_state[0]
            shaped_reward = reward - POSITION_PENALTY_COEF * abs(cart_position)

            done = terminated or truncated

            rollout_states.append(state)
            rollout_actions.append(action.item())
            rollout_rewards.append(shaped_reward)
            rollout_next_states.append(next_state)
            rollout_terminated.append(terminated)
            rollout_episode_ended.append(done)
            rollout_old_log_probs.append(old_log_prob.item())

            if done:
                reward_history.append(current_episode_reward)
                episode_rewards.append(current_episode_reward)
                episode_completed += 1

                current_episode_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state
        
        states = torch.as_tensor(rollout_states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(rollout_actions, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(rollout_rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(rollout_next_states, dtype=torch.float32, device=device)
        terminated_flags = torch.as_tensor(rollout_terminated, dtype=torch.float32, device=device)
        episode_ended_flags = torch.as_tensor(rollout_episode_ended, dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(rollout_old_log_probs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, old_values = model(states)
            _, next_values = model(next_states)

            old_values = old_values.squeeze(-1)
            next_values = next_values.squeeze(-1)

            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, device=device)

            # Compute the Generalized Advantage Estimation (GAE) in reverse order 
            # Because we need to compute the advantage for each time step based on the future rewards and values
            for t in reversed(range(ROLLOUT_STEPS)):
                bootstrap_value = 1.0 - terminated_flags[t]  # If the episode ended, we don't bootstrap

                td_error = (
                    rewards[t]
                    + GAMMA 
                    * bootstrap_value
                    * next_values[t]
                    - old_values[t]
                )

                continuation_mask = 1.0 - episode_ended_flags[t]  # If the episode ended, we don't continue the GAE

                gae = (
                    td_error
                    + GAMMA 
                    * GAE_LAMBDA 
                    * continuation_mask 
                    * gae
                )

                advantages[t] = gae
            
            returns = advantages + old_values

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for epoch in range(PPO_EPOCHS):
            logits, predicted_values = model(states)
            distribution = Categorical(logits=logits)

            new_log_probs = distribution.log_prob(actions)
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            unclipped_objective = ratio * normalized_advantages

            clipped_ratio = torch.clamp(
                ratio, 
                1.0 - CLIP_EPSILON, 
                1.0 + CLIP_EPSILON
            )

            clipped_objective = clipped_ratio * normalized_advantages

            actor_loss_clipped = -torch.min(unclipped_objective, clipped_objective).mean()

            critic_loss = 0.5 * (returns - predicted_values.squeeze(-1)).pow(2).mean()

            entropy = distribution.entropy().mean()

            total_loss = (
                actor_loss_clipped
                + VALUE_COEF * critic_loss
                - ENTROPY_COEF * entropy
            )

            optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            clip_fraction = (
                (ratio - 1.0).abs() > CLIP_EPSILON
            ).float().mean()

            print(
                f"Epoch {epoch + 1} | "
                f"Ratio mean {ratio.mean().item():.4f} | "
                f"Ratio min {ratio.min().item():.4f} | "
                f"Ratio max {ratio.max().item():.4f} | "
                f"Clip fraction {clip_fraction.item():.4f}"
            )

        average_reward = (
            sum(reward_history) / len(reward_history) if reward_history else 0.0
        )

        update_history["average_reward"].append(average_reward)
        update_history["actor_loss"].append(actor_loss_clipped.item())
        update_history["critic_loss"].append(critic_loss.item())
        update_history["entropy"].append(entropy.item())
        update_history["clip_fraction"].append(clip_fraction.item())
        update_history["ratio_mean"].append(ratio.mean().item())

        print(
            f"Update {update:4d} | "
            f"Episodes {episode_completed:4d} | "
            f"Average reward {average_reward:7.2f} | "
            f"Actor loss {actor_loss_clipped.item():8.4f} | "
            f"Critic loss {critic_loss.item():8.4f}"
        )

    env.close()

    plot_training_curves(episode_rewards, update_history)

    # Save the trained model into model folder
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), f"model/ppo_{ENV_NAME}_model.pth")

    # Save the training configuration into a text file
    with open(f"model/ppo_{ENV_NAME}_config.txt", "w") as f:
        f.write(
            f"Environment: {ENV_NAME}\n"
            f"Seed: {SEED}\n"
            f"Gamma: {GAMMA}\n"
            f"Learning rate: {LEARNING_RATE}\n"
            f"Value coefficient: {VALUE_COEF}\n"
            f"Entropy coefficient: {ENTROPY_COEF}\n"
            f"Position penalty coefficient: {POSITION_PENALTY_COEF}\n"
            f"Max episodes: {MAX_EPISODES}\n"
            f"Hidden dimension: {HIDDEN_DIM}\n"
            f"Rollout steps: {ROLLOUT_STEPS}\n"
            f"Max gradient norm: {MAX_GRAD_NORM}\n"
            f"GAE lambda: {GAE_LAMBDA}\n"
            f"Clip epsilon: {CLIP_EPSILON}\n"
            f"PPO epochs: {PPO_EPOCHS}\n"
        )

if __name__ == "__main__":
    train()
    
