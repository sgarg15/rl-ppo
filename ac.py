import torch
import torch.nn as nn


class ActorCriticAgent(nn.Module):
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
