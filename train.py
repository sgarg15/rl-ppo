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
# for CartPole's short episodes doesn't transfer well to LunarLander and BipedalWalker.
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
    """
    Train a PPO agent on the specified Gymnasium environment.
    The training configuration is determined by the ENV_PRESETS dictionary.
    """

    # Load environment-specific hyperparameters
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

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the Gymnasium environment and set the random seed for reproducibility
    env = gym.make(env_name, render_mode="human" if render else None)
    env.action_space.seed(SEED)

    # Get the observation dimension from the environment's observation space
    observation_dim = env.observation_space.shape[0]

    # Create the actor-critic model and move it to the specified device (GPU or CPU)
    model = create_model(env, observation_dim, hidden_dim).to(device)

    # Create the optimizer for training the model using Adam with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the environment and variables for tracking rewards and episodes
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

    # Main training loop for the specified number of episodes/updates
    for update in range(num_updates):
        # Initialize lists to store rollout data for the current update
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_states = []
        rollout_terminated = []
        rollout_episode_ended = []
        rollout_old_log_probs = []

        # Collect rollout data for the specified number of steps
        for step in range(rollout_steps):
            # Convert the current state to a tensor and add a batch dimension
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # The torch.no_grad() context is used to avoid computing gradients during action selection
            # Get the action, policy action, log probability of the action, and state value from the model
            with torch.no_grad():
                env_action, policy_action, old_log_prob, _ = model.act(state_tensor)

            # Step the environment using the selected action and receive the next state, reward, and done flags
            next_state, reward, terminated, truncated, _ = env.step(
                to_env_action(env_action, env.action_space)
            )

            # Update the current episode reward with the received reward for logging purposes
            current_episode_reward += reward

            # Apply position penalty if specified in the environment preset
            if position_penalty_coef > 0:
                shaped_reward = reward - position_penalty_coef * abs(next_state[0])
            else:
                shaped_reward = reward

            # As mentioned in the readme, due to the nature of the environments, we scale the rewards to stabilize training
            shaped_reward *= reward_scale

            done = terminated or truncated

            # Store the collected rollout data for the current step to be used for training the model later
            rollout_states.append(state)
            rollout_actions.append(to_stored_action(policy_action, env.action_space))
            rollout_rewards.append(shaped_reward)
            rollout_next_states.append(next_state)
            rollout_terminated.append(terminated)
            rollout_episode_ended.append(done)
            rollout_old_log_probs.append(old_log_prob.item())

            # If the episode is done, reset the environment and log the episode reward; otherwise, continue to the next state
            if done:
                reward_history.append(current_episode_reward)
                episode_rewards.append(current_episode_reward)
                episode_completed += 1

                current_episode_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state

        # Convert the collected rollout data into PyTorch tensors for training
        states = torch.as_tensor(np.array(rollout_states), dtype=torch.float32, device=device)
        action_dtype = torch.int64 if isinstance(env.action_space, gym.spaces.Discrete) else torch.float32
        actions = torch.as_tensor(np.array(rollout_actions), dtype=action_dtype, device=device)
        rewards = torch.as_tensor(rollout_rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(rollout_next_states), dtype=torch.float32, device=device)
        terminated_flags = torch.as_tensor(rollout_terminated, dtype=torch.float32, device=device)
        episode_ended_flags = torch.as_tensor(rollout_episode_ended, dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(rollout_old_log_probs, dtype=torch.float32, device=device)

        # Compute the advantages and returns using Generalized Advantage Estimation (GAE) for the collected rollout data
        with torch.no_grad():
            # Get the state values for the current and next states from the model to compute the advantages and returns
            old_values = model.get_value(states)
            next_values = model.get_value(next_states)

            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, device=device)

            # Compute the Generalized Advantage Estimation (GAE) in reverse order 
            # Because we need to compute the advantage for each time step based on the future rewards and values
            for t in reversed(range(rollout_steps)):
                bootstrap_value = 1.0 - terminated_flags[t]  # If the episode ended, we don't bootstrap

                # Compute the temporal difference (TD) error for the current time step based on the reward, next value, and old value
                td_error = (
                    rewards[t]
                    + GAMMA 
                    * bootstrap_value
                    * next_values[t]
                    - old_values[t]
                )

                # This mask is used to determine whether to continue the GAE computation based on whether the episode has ended or not.
                # If the episode has ended, we don't continue the GAE computation for that time step.
                continuation_mask = 1.0 - episode_ended_flags[t]  # If the episode ended, we don't continue the GAE

                # Update the GAE for the current time step based on the TD error, discount factor, lambda parameter, and continuation mask
                gae = (
                    td_error
                    + GAMMA 
                    * GAE_LAMBDA 
                    * continuation_mask 
                    * gae
                )

                # Store the computed GAE for the current time step in the advantages tensor
                advantages[t] = gae
        
            # Compute the returns for the collected rollout data by adding the advantages to the old state values
            returns = advantages + old_values

        # Normalize the advantages to have zero mean and unit variance for stable training
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Update the model using the collected rollout data and computed advantages/returns for the specified number of PPO epochs
        for epoch in range(PPO_EPOCHS):

            # Evaluate the log probabilities, entropy, and predicted state values for the collected rollout states and actions using the current model
            new_log_probs, entropy_per_sample, predicted_values = model.evaluate_actions(states, actions)

            # The log ratio is computed as the difference between the new log probabilities and the old log probabilities for the actions taken during the rollout.
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            # Compute the unclipped and clipped objectives for the PPO loss function based on the ratio of new to old probabilities and the normalized advantages.
            unclipped_objective = ratio * normalized_advantages

            # Clip the ratio in PPO to prevent large policy updates and ensure stable training
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON
            )

            # The clipped objective allows for a more conservative update to the policy by limiting the change in action probabilities, which helps prevent divergence during training.
            clipped_objective = clipped_ratio * normalized_advantages

            # Compute the actor loss using the minimum of the unclipped and clipped objectives to ensure that the policy update is conservative and does not deviate too much from the old policy.
            actor_loss_clipped = -torch.min(unclipped_objective, clipped_objective).mean()

            # The critic loss allows the critic to learn to predict the state values accurately by minimizing the mean squared error between the predicted values and the computed returns.
            critic_loss = 0.5 * (returns - predicted_values).pow(2).mean()

            entropy = entropy_per_sample.mean()

            # The total loss is a combination of the actor loss, critic loss, and entropy regularization term. 
            # The actor loss encourages the policy to improve, the critic loss ensures accurate value predictions, 
            # and the entropy term promotes exploration by encouraging a more diverse action distribution.
            total_loss = (
                actor_loss_clipped
                + value_coef * critic_loss
                - entropy_coef * entropy
            )

            # Zero out the gradients of the model parameters
            optimizer.zero_grad()
            # Compute the gradients of the total loss with respect to the model parameters using backpropagation
            total_loss.backward()

            # Clip the gradients to prevent exploding gradients and ensure stable training.
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Update the model parameters using the optimizer based on the computed gradients
            # The backward pass computes the gradients and stores them directly in the model parameters, and the optimizer.step() updates the parameters based on those gradients.
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


