from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

ENV_NAME = "CartPole-v1"
SEED = 42

GAMMA = 0.99 # Discount factor for future rewards
LEARNING_RATE = 0.01 # Learning rate for the optimizer

VALUE_COEF = 0.5 # Coefficient for the value loss
ENTROPY_COEF = 0.001 # Coefficient for the entropy loss

MAX_EPISODES = 1000 # Maximum number of episodes to train
HIDDEN_DIM = 128 # Number of hidden units in the neural network

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network architecture for reinforcement learning.

    The reason for using a shared architecture is to allow the actor and critic networks to share some common features, which can lead to better generalization and more efficient learning. 
    The shared layers extract features from the input observations, which are then used by both the actor and critic networks to make decisions and evaluate states, respectively.
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
    
        # Shared layers between the actor and critic networks
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor network: outputs the action probabilities
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic network: outputs the state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_layers(observation)

        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)  # Remove the last dimension for state value

        return action_logits, state_value
    

def train() -> None:
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME, render_mode="human")
    env.action_space.seed(SEED)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(observation_dim, action_dim, HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    recent_rewards = deque(maxlen=100)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=SEED if episode == 1 else None)

        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            action_logits, state_value = model(state_tensor)

            # Create a categorical distribution over the action logits
            distribution = Categorical(logits=action_logits)

            # Sample an action from the distribution and compute its log probability and compute the entropy of the distribution
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated
            episode_reward += reward

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                # Compute the value of the next state using the critic network
                _, next_state_value = model(next_state_tensor)

                # If the episode is terminated, the next state value is zero
                if terminated:
                    next_state_value = torch.zeros_like(next_state_value)
                
                # Compute the TD target for the critic network using the Bellman equation which is the reward plus the discounted value of the next state
                td_target = (
                    torch.tensor([reward], dtype=torch.float32) 
                    + GAMMA * next_state_value
                )

            # Compute the advantage for the actor network
            # A_t = R_t + gamma * V(s_{t+1}) - V(s_t)
            advantage = td_target - state_value  

            # Actor performs gradient ascent on:
            # log pi(a_t | s_t) * A_t
            # PyTorch minimizes losses, so we negate it.
            # We detach the advantage to prevent gradients from flowing into the critic network during the actor update.
            # And mean over the batch (in this case, a single sample) to get a scalar loss value.
            actor_loss = -(log_prob * advantage.detach()).mean()

            critic_loss = advantage.pow(2).mean()  # Mean squared error loss for the critic

            entropy_bonus = entropy.mean()

            total_loss = (
                actor_loss
                + VALUE_COEF * critic_loss
                - ENTROPY_COEF * entropy_bonus
            )

            optimizer.zero_grad()
            # Compute gradients for the total loss 
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping to prevent exploding gradients

            optimizer.step()

            state = next_state

        recent_rewards.append(episode_reward)
        average_reward = sum(recent_rewards) / len(recent_rewards)

        if episode % 10 == 0:
            print(
                        f"Episode: {episode:4d} | "
                        f"Reward: {episode_reward:6.1f} | "
                        f"Average(100): {average_reward:6.1f}"
                    )
        
        if len(recent_rewards) == 100 and average_reward >= 475.0:
            print(f"Solved in {episode} episodes!")
            print(f"Average reward over the last 100 episodes: {average_reward:.2f}")
            break

    env.close()

if __name__ == "__main__":
    train()
    
