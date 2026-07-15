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
        features = self.shared_layers(observations)

        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)

        return action_logits, state_value

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)

        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)

        return action, action, log_prob, values

    def evaluate_actions(self, observations: torch.Tensor, policy_actions: torch.Tensor):
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)

        log_probs = distribution.log_prob(policy_actions.long())
        entropy = distribution.entropy()

        return log_probs, entropy, values

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
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

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_layers(observations)

        action_mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
        action_std = torch.exp(log_std).expand_as(action_mean)

        state_value = self.critic(features).squeeze(-1)

        return action_mean, action_std, state_value

    @staticmethod
    def _log_prob(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        # tanh-squashing change-of-variables correction, summed across action dims.
        squashed_actions = torch.tanh(raw_actions)
        log_probs = distribution.log_prob(raw_actions) - torch.log(1.0 - squashed_actions.pow(2) + 1e-6)
        return log_probs.sum(dim=-1)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        means, stds, values = self.forward(observations)
        distribution = Normal(means, stds)

        raw_action = means if deterministic else distribution.sample()
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
    if isinstance(env.action_space, gym.spaces.Discrete):
        return DiscreteActorCritic(observation_dim, env.action_space.n, hidden_dim)

    if isinstance(env.action_space, gym.spaces.Box):
        if len(env.action_space.shape) != 1:
            raise ValueError("Only one-dimensional continuous action spaces are supported.")

        return ContinuousActorCritic(observation_dim, env.action_space.shape[0], hidden_dim)

    raise TypeError(f"Unsupported action space: {env.action_space}")


def to_env_action(action: torch.Tensor, action_space: gym.Space):
    """Convert a batch-of-1 model action tensor into what env.step() expects."""
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action.item())

    return action.squeeze(0).cpu().numpy()


def to_stored_action(action: torch.Tensor, action_space: gym.Space):
    """Convert a batch-of-1 policy action tensor into what the rollout buffer stores."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action.item()

    return action.squeeze(0).cpu().numpy()
