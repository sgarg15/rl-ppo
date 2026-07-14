import gymnasium as gym
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

def train() -> None:
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME, render_mode="human")
    env.action_space.seed(SEED)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticAgent(observation_dim, action_dim, HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    state, _ = env.reset(seed=SEED)
    current_episode_reward = 0.0
    reward_history = deque(maxlen=100)
    episode_completed = 0

    for update in range(MAX_EPISODES):
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_next_states = []
        rollout_terminated = []
        rollout_episode_ended = []
        rollout_old_log_probs = []

        for step in range(ROLLOUT_STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

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
                episode_completed += 1

                current_episode_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state
        
        states = torch.as_tensor(rollout_states, dtype=torch.float32)
        actions = torch.as_tensor(rollout_actions, dtype=torch.int64)
        rewards = torch.as_tensor(rollout_rewards, dtype=torch.float32)
        next_states = torch.as_tensor(rollout_next_states, dtype=torch.float32)
        terminated_flags = torch.as_tensor(rollout_terminated, dtype=torch.float32)
        episode_ended_flags = torch.as_tensor(rollout_episode_ended, dtype=torch.float32)
        old_log_probs = torch.as_tensor(rollout_old_log_probs, dtype=torch.float32)

        with torch.no_grad():
            _, old_values = model(states)
            _, next_values = model(next_states)

            old_values = old_values.squeeze(-1)
            next_values = next_values.squeeze(-1)

            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0)

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

        # logits, predicted_values = model(states)

        # distribution = Categorical(logits=logits)
        # log_probs = distribution.log_prob(actions)

        # # actor_loss = -(log_probs * normalized_advantages).mean()
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

            total_loss = (
                actor_loss_clipped
                + VALUE_COEF * critic_loss
                - ENTROPY_COEF * distribution.entropy().mean()
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

        print(
            f"Update {update:4d} | "
            f"Episodes {episode_completed:4d} | "
            f"Average reward {average_reward:7.2f} | "
            f"Actor loss {actor_loss_clipped.item():8.4f} | "
            f"Critic loss {critic_loss.item():8.4f}"
        )

    env.close()

if __name__ == "__main__":
    train()
    
