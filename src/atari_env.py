import ale_py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation

gym.register_envs(ale_py)


def make_env(game_id: str, render_mode: str, obs_type: str):
    # env = gym.make("ALE/Pong-v5",render_mode="rgb_array", obs_type="grayscale")
    env = gym.make(id=game_id, render_mode=render_mode, obs_type=obs_type)
    # Resize the image to 84x84 to match playing atari game
    env = ResizeObservation(env, shape=(84, 84))
    # Stack 4 consecutive frames (this uses a sliding window)
    env = FrameStackObservation(env, stack_size=4)
    return env
