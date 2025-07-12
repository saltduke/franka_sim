from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

# from gymnasium.envs.registration import register
from gymnasium.envs.registration import register
import gymnasium
register(
    id="PandaSetPin-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
#register(
#    id="PandaInsertPegVision-v0",
#    entry_point="franka_sim.envs:PandaInsertPegGymEnv",
#   max_episode_steps=100,
#    kwargs={"image_obs": True},
#)
