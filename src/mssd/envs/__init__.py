from .cliff_wrapper import CliffWalkingContinuousObs
from .cartpole_wrapper import CartPoleLogged
from .shifts import ShiftInjector, ShiftConfig


def make_env(env_cfg: dict):
    """Factory: create a wrapped Gymnasium environment from config."""
    import gymnasium as gym

    env = gym.make(env_cfg["name"])
    wrapper = env_cfg.get("wrapper", None)
    if wrapper == "CliffWalkingContinuousObs":
        env = CliffWalkingContinuousObs(env)
    elif wrapper == "CartPoleLogged":
        env = CartPoleLogged(env)
    return env
