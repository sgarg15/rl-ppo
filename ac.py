from abc import ABC, abstractmethod

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class BaseActorCritic(nn.Module, ABC):
    """
    Common interface so the PPO training loop does not need to know whether
    the policy is discrete or continuous.
    """

    @abstractmethod
    def act(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (env_action, policy_action, log_prob, value)."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, observations: torch.Tensor, policy_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_prob, entropy, value) for previously stored policy_actions."""
        raise NotImplementedError

    @abstractmethod
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiscreteActorCritic(BaseActorCritic):
    """
    Actor-critic for Discrete action spaces (e.g. CartPole, LunarLander).

    The shared layers extract features from the input observations, which are
    then used by both the actor and critic to make decisions and evaluate
    states, respectively.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        """
        Initialize the DiscreteActorCritic model with shared layers, actor, and critic.
        """
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared layers, actor, and critic.
        Returns the action logits and state value for the given observations.
        """
        
        # Pass the observations through the shared layers to extract features
        features = self.shared_layers(observations)

        # Pass the features through the final layers of the action and critic networks to get the action logits and state value
        action_logits = self.actor(features)

        # Squeeze the last dimension of the critic output to get a 1D tensor for state value
        state_value = self.critic(features).squeeze(-1)

        return action_logits, state_value

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        """
        Select an action based on the current policy given the observations.
        If deterministic is True, select the action with the highest probability.
        Otherwise, sample an action from the policy distribution.
        Returns the selected action, policy action, log probability of the action, and state value.
        """

        # Forward pass to get action logits and state value given the state observations
        logits, values = self.forward(observations)
        # Create a categorical distribution based on the action logits to be able to sample actions and compute log probabilities
        distribution = Categorical(logits=logits)

        # Select the action based ont the flag for deterministic or stochastic
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        # Compute the log probability of the selected action using the distribution to be used in the PPO loss function
        log_prob = distribution.log_prob(action)

        return action, action, log_prob, values

    def evaluate_actions(self, observations: torch.Tensor, policy_actions: torch.Tensor):
        """
        Evaluate the log probability, entropy, and state value for the given observations and policy actions.
        This is used during the PPO update to compute the loss and update the model parameters.
        """
        # Forward pass to get action logits and state value given the state observations
        logits, values = self.forward(observations)
        # Similar to the act method, create a categorical distribution based on the action logits
        distribution = Categorical(logits=logits)

        # Compute the log probability of the given policy actions using the distribution
        log_probs = distribution.log_prob(policy_actions.long())
        # Compute the entropy of the distribution to encourage exploration during training
        entropy = distribution.entropy()

        return log_probs, entropy, values

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get the state value for the given observations by forwarding through the model.
        """
        _, values = self.forward(observations)
        return values


class ContinuousActorCritic(BaseActorCritic):
    """
    Actor-critic for Box action spaces whose bounds are [-1, 1] (e.g.
    BipedalWalker). Actions are sampled from a Normal distribution and
    squashed with tanh; env-facing environments with other bounds need
    rescaling outside this class.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # The actor_mean layer outputs the mean of the action distribution
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # The actor_log_std parameter represents the log standard deviation of the action distribution. The critic layer outputs the state value.
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as DiscreteActorCritic.forward, but returns action mean and std for Normal distribution instead of logits.
        """
        features = self.shared_layers(observations)

        action_mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
        action_std = torch.exp(log_std).expand_as(action_mean)

        state_value = self.critic(features).squeeze(-1)

        return action_mean, action_std, state_value

    @staticmethod
    def _log_prob(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        The point of this function is to compute the log probability of the raw actions sampled from the Normal distribution, taking into account the tanh-squashing change-of-variables correction.
        So that we can compute the log probability of the actions after they have been squashed by the tanh function, which is necessary for the PPO loss function.
        """
        squashed_actions = torch.tanh(raw_actions)
        log_probs = distribution.log_prob(raw_actions) - torch.log(1.0 - squashed_actions.pow(2) + 1e-6)
        return log_probs.sum(dim=-1)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        """
        Select an action based on the current policy given the observations.
        If deterministic is True, select the action with the highest probability (mean).
        Otherwise, sample an action from the policy distribution.
        This is different from the DiscreteActorCritic.act method because the action space is continuous and requires sampling from a Normal distribution.
        Returns the selected action, policy action, log probability of the action, and state value.
        """
        means, stds, values = self.forward(observations)
        distribution = Normal(means, stds)

        # The raw_action is sampled from the Normal distribution if deterministic is False, otherwise it is set to the mean of the distribution. 
        raw_action = means if deterministic else distribution.sample()
        # The env_action is obtained by applying the tanh function to the raw_action to ensure that it lies within the bounds of the action space. 
        # The log_prob is computed using the _log_prob method, which takes into account the tanh-squashing change-of-variables correction.
        env_action = torch.tanh(raw_action)
        log_prob = self._log_prob(distribution, raw_action)

        return env_action, raw_action, log_prob, values

    def evaluate_actions(self, observations: torch.Tensor, policy_actions: torch.Tensor):
        means, stds, values = self.forward(observations)
        distribution = Normal(means, stds)

        log_probs = self._log_prob(distribution, policy_actions)
        # Approximation: entropy of the underlying Normal, ignoring the tanh squash.
        entropy = distribution.entropy().sum(dim=-1)

        return log_probs, entropy, values

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        _, _, values = self.forward(observations)
        return values


def create_model(env: gym.Env, observation_dim: int, hidden_dim: int) -> BaseActorCritic:
    """
    Create an actor-critic model based on the environment's action space type.
    """
    
    # If its a discrete action space, create a DiscreteActorCritic model with the specified observation and action dimensions, and hidden layer size.
    if isinstance(env.action_space, gym.spaces.Discrete):
        return DiscreteActorCritic(observation_dim, env.action_space.n, hidden_dim)

    # If its a continuous action space, create a ContinuousActorCritic model with the specified observation and action dimensions, and hidden layer size.
    if isinstance(env.action_space, gym.spaces.Box):
        if len(env.action_space.shape) != 1:
            raise ValueError("Only one-dimensional continuous action spaces are supported.")

        return ContinuousActorCritic(observation_dim, env.action_space.shape[0], hidden_dim)

    raise TypeError(f"Unsupported action space: {env.action_space}")


def to_env_action(action: torch.Tensor, action_space: gym.Space):
    """Convert a batch-of-1 model action tensor into what env.step() expects."""
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action.item())

    # For continuous action spaces, we need to convert the action tensor into a NumPy array and ensure it has the correct shape for the environment.
    return action.squeeze(0).cpu().numpy()


def to_stored_action(action: torch.Tensor, action_space: gym.Space):
    """Convert a batch-of-1 policy action tensor into what the rollout buffer stores."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action.item()

    return action.squeeze(0).cpu().numpy()
