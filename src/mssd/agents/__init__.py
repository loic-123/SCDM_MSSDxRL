from .tabular_q import TabularQAgent
from .dqn import DQNAgent
from ..utils.seeding import resolve_device


def make_agent(agent_cfg: dict):
    """Factory: create an RL agent from config."""
    agent_type = agent_cfg["type"]
    if agent_type == "tabular_q":
        return TabularQAgent(
            lr=agent_cfg.get("lr", 0.1),
            gamma=agent_cfg.get("gamma", 0.99),
            epsilon=agent_cfg.get("epsilon", 0.1),
        )
    elif agent_type == "dqn":
        device = resolve_device(agent_cfg.get("device", "auto"))
        return DQNAgent(
            lr=agent_cfg.get("lr", 1e-3),
            gamma=agent_cfg.get("gamma", 0.99),
            epsilon_start=agent_cfg.get("epsilon_start", 1.0),
            epsilon_end=agent_cfg.get("epsilon_end", 0.01),
            epsilon_decay=agent_cfg.get("epsilon_decay", 500),
            buffer_size=agent_cfg.get("buffer_size", 10000),
            batch_size=agent_cfg.get("batch_size", 64),
            target_update_freq=agent_cfg.get("target_update_freq", 10),
            hidden=agent_cfg.get("hidden", 128),
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
