"""Train and save an RL agent."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.utils.config import load_config
from mssd.utils.seeding import set_all_seeds, resolve_device
from mssd.envs import make_env
from mssd.agents import make_agent


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--config", required=True, help="Path to env config YAML")
    parser.add_argument("--output", required=True, help="Path to save agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device (auto/cuda/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(args.seed)

    # Allow CLI device override
    if args.device:
        cfg["agent"]["device"] = args.device

    device = resolve_device(cfg["agent"].get("device", "auto"))
    print(f"Training on device: {device}")

    env = make_env(cfg["env"])
    agent = make_agent(cfg["agent"])
    metrics = agent.train(env, num_episodes=cfg["agent"]["train_episodes"])
    agent.save(args.output)

    rewards = metrics["episode_rewards"]
    avg = sum(rewards[-100:]) / min(len(rewards), 100)
    print(f"Training complete. Final 100-ep avg reward: {avg:.1f}")
    print(f"Agent saved to: {args.output}")


if __name__ == "__main__":
    main()
