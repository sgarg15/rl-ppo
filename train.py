import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from ac import create_model, to_env_action, to_stored_action
from plotting import plot_training_curves

SEED = 42

GAMMA = 0.99 # Discount factor for future rewards

MAX_GRAD_NORM = 0.5 # Maximum gradient norm for gradient clipping

GAE_LAMBDA = 0.95 # Lambda parameter for Generalized Advantage Estimation (GAE)

CLIP_EPSILON = 0.2 # Clipping parameter for PPO

PPO_EPOCHS = 4 # Number of epochs to update the model per rollout

# Per-environment overrides, since a rollout/learning-rate/hidden-size tuned
# for CartPole's short episodes doesn't transfer well to LunarLander.
ENV_PRESETS = {
    "CartPole-v1": {
        "learning_rate": 0.01,
        "rollout_steps": 256,
        "num_updates": 1000,
        "hidden_dim": 128,
        "entropy_coef": 0.001,
        "position_penalty_coef": 0.1, # Penalizes distance of the cart from the center
        "reward_scale": 1.0,
        "value_coef": 0.5,
    },
    "LunarLander-v3": {
        # 1e-4/0.25 was tried to stabilize late-stage training but caused the
        # policy to freeze at ~40 avg reward (ratio stuck at 1.0, no further
        # updates); 3e-4/0.5 reached 190 avg at 2000 updates, so keep it.
        "learning_rate": 3e-4,
        "rollout_steps": 2048,
        "num_updates": 3000, # Reward was still climbing at 2000 updates (reached ~190 avg)
        "hidden_dim": 128,
        "entropy_coef": 0.01,
        "position_penalty_coef": 0.0,
        # LunarLander returns can be in the hundreds; without scaling, critic_loss
        # dwarfs actor_loss and the shared gradient-norm clip crushes actor updates.
        "reward_scale": 0.1,
        "value_coef": 0.5,
    },
    # Continuous-action environment (Box action space), exercising ContinuousActorCritic.
    "BipedalWalker-v3": {
        "learning_rate": 3e-4,
        "rollout_steps": 2048,
        "num_updates": 3000,
        "hidden_dim": 256,
        "entropy_coef": 0.0,
        "position_penalty_coef": 0.0,
        "reward_scale": 1.0,
        "value_coef": 0.5,
    },
}

def train(env_name: str, render: bool) -> None:
    preset = ENV_PRESETS[env_name]
    learning_rate = preset["learning_rate"]
    rollout_steps = preset["rollout_steps"]
    num_updates = preset["num_updates"]
    hidden_dim = preset["hidden_dim"]
    entropy_coef = preset["entropy_coef"]
    position_penalty_coef = preset["position_penalty_coef"]
    reward_scale = preset["reward_scale"]
    value_coef = preset["value_coef"]

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(env_name, render_mode="human" if render else None)
    env.action_space.seed(SEED)

    observation_dim = env.observation_space.shape[0]

    model = create_model(env, observation_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    for update in range(num_updates):
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_states = []
        rollout_terminated = []
        rollout_episode_ended = []
        rollout_old_log_probs = []

        for step in range(rollout_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                env_action, policy_action, old_log_prob, _ = model.act(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(
                to_env_action(env_action, env.action_space)
            )

            current_episode_reward += reward

            if position_penalty_coef > 0:
                shaped_reward = reward - position_penalty_coef * abs(next_state[0])
            else:
                shaped_reward = reward

            shaped_reward *= reward_scale

            done = terminated or truncated

            rollout_states.append(state)
            rollout_actions.append(to_stored_action(policy_action, env.action_space))
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

        states = torch.as_tensor(np.array(rollout_states), dtype=torch.float32, device=device)
        action_dtype = torch.int64 if isinstance(env.action_space, gym.spaces.Discrete) else torch.float32
        actions = torch.as_tensor(np.array(rollout_actions), dtype=action_dtype, device=device)
        rewards = torch.as_tensor(rollout_rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(rollout_next_states), dtype=torch.float32, device=device)
        terminated_flags = torch.as_tensor(rollout_terminated, dtype=torch.float32, device=device)
        episode_ended_flags = torch.as_tensor(rollout_episode_ended, dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(rollout_old_log_probs, dtype=torch.float32, device=device)

        with torch.no_grad():
            old_values = model.get_value(states)
            next_values = model.get_value(next_states)

            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, device=device)

            # Compute the Generalized Advantage Estimation (GAE) in reverse order 
            # Because we need to compute the advantage for each time step based on the future rewards and values
            for t in reversed(range(rollout_steps)):
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
            new_log_probs, entropy_per_sample, predicted_values = model.evaluate_actions(states, actions)

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

            critic_loss = 0.5 * (returns - predicted_values).pow(2).mean()

            entropy = entropy_per_sample.mean()

            total_loss = (
                actor_loss_clipped
                + value_coef * critic_loss
                - entropy_coef * entropy
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

    plot_training_curves(env_name, episode_rewards, update_history)

    # Save the trained model into model folder
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), f"model/ppo_{env_name}_model.pth")

    # Save the training configuration into a text file
    with open(f"model/ppo_{env_name}_config.txt", "w") as f:
        f.write(
            f"Environment: {env_name}\n"
            f"Seed: {SEED}\n"
            f"Gamma: {GAMMA}\n"
            f"Learning rate: {learning_rate}\n"
            f"Value coefficient: {value_coef}\n"
            f"Entropy coefficient: {entropy_coef}\n"
            f"Position penalty coefficient: {position_penalty_coef}\n"
            f"Reward scale: {reward_scale}\n"
            f"Num updates: {num_updates}\n"
            f"Hidden dimension: {hidden_dim}\n"
            f"Rollout steps: {rollout_steps}\n"
            f"Max gradient norm: {MAX_GRAD_NORM}\n"
            f"GAE lambda: {GAE_LAMBDA}\n"
            f"Clip epsilon: {CLIP_EPSILON}\n"
            f"PPO epochs: {PPO_EPOCHS}\n"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent")
    parser.add_argument(
        "--env",
        choices=list(ENV_PRESETS),
        default="CartPole-v1",
        help="Gymnasium environment to train on",
    )
    parser.add_argument("--render", action="store_true", help="Render the environment while training")
    args = parser.parse_args()

    train(env_name=args.env, render=args.render)


