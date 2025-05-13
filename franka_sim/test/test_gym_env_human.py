import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import turtle

env = envs.PandaPickCubeGymEnv(action_scale=(0.1, 1), render_mode="human")
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


m = env.model
d = env.data

reset = False
KEY_SPACE = 32


def key_callback(keycode):
    if keycode == KEY_SPACE:
        global reset
        reset = True

window = turtle.Screen()
startposition = np.array([0,   0, 0, 0, 0, 0, 0.01 ])
global position
position = startposition
def go_f():
    position[0] = position[0] + 0.01
    print(position)
def go_b():
    position[0] = position[0] - 0.01
    print(position)
def go_l():
    position[1] = position[1] + 0.01
    print(position)
def go_r():
    position[1] = position[1] - 0.01
    print(position)
def go_u():
    position[2] = position[2] + 0.01
    print(position)
def go_d():
    position[2] = position[2] - 0.01
    print(position)
def grap():
    position[3] = position[3] + 0.01
    print(position)
def letgo():
    position[3] = position[3] - 0.01
    print(position)
window.onkeypress(go_f, "w")
window.onkeypress(go_b, "s")
window.onkeypress(go_l, "a")
window.onkeypress(go_r, "d")
window.onkeypress(go_u, "q")
window.onkeypress(go_d, "e")
window.onkeypress(grap, "y")
window.onkeypress(letgo, "x")
window.listen()
env.reset()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start = time.time()
    while viewer.is_running():
        if reset:
            env.reset()
            reset = False
            position = startposition
        else:
            window.update()
            step_start = time.time()
            env.step(position)
            viewer.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

