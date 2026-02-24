"""Collect reference observations from a trained agent in the nominal environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.utils.config import load_config
from mssd.utils.seeding import set_all_seeds
from mssd.envs import make_env
from mssd.agents import make_agent
from mssd.monitor.reference_buffer import ReferenceBuffer


def main():
    parser = argparse.ArgumentParser(description="Collect reference observations")
    parser.add_argument("--config", required=True, help="Path to env config YAML")
    parser.add_argument("--agent", required=True, help="Path to saved agent")
    parser.add_argument("--output", required=True, help="Path to save reference .npz")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(args.seed)

    env = make_env(cfg["env"])
    agent = make_agent(cfg["agent"])
    agent.load(args.agent)

    buffer = ReferenceBuffer()
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            buffer.add(obs)
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    buffer.save(args.output)
    print(f"Collected {len(buffer)} reference observations from {args.n_episodes} episodes.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
