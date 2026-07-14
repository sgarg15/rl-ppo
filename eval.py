import argparse

import gymnasium as gym
import torch
from torch.distributions import Categorical

from ac import ActorCriticAgent
from train import ENV_PRESETS


def evaluate(env_name: str, episodes: int, render: bool, deterministic: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = ENV_PRESETS[env_name]["hidden_dim"]
    model_path = f"model/ppo_{env_name}_model.pth"

    env = gym.make(env_name, render_mode="human" if render else None)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticAgent(observation_dim, action_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action_logits, _ = model(state_tensor)

                if deterministic:
                    action = action_logits.argmax(dim=-1)
                else:
                    action = Categorical(logits=action_logits).sample()

            state, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1:3d} | Reward {episode_reward:7.2f}")

    env.close()

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage reward over {episodes} episodes: {average_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument(
        "--env",
        choices=list(ENV_PRESETS),
        default="CartPole-v1",
        help="Gymnasium environment to evaluate on",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of taking the argmax",
    )
    args = parser.parse_args()

    evaluate(
        env_name=args.env,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
    )
