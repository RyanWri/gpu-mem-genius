import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py)


def make_env(game_id: str, render_mode: str, obs_type: str):
    env = gym.make(id=game_id, frameskip=1)
    # Apply AtariPreprocessing to add frame skipping Resize the image to 84x84 to match playing atari game
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
    )
    return env
