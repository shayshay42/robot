# train_sac_ur5e_lift.py
import os
os.environ["MUJOCO_GL"] = "glfw"  # Set before importing robosuite

import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
import numpy as np

def make_env():
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=False,           # headless for speed
        has_offscreen_renderer=False, # disable offscreen rendering that's causing issues
        use_camera_obs=False,         # low-dim obs
        control_freq=20,
        horizon=400,
        reward_shaping=True,          # denser reward -> faster learning
    )
    env = GymWrapper(env)            # flattens obs, Gym API
    env = TimeLimit(env, max_episode_steps=400)
    return env

# parallel envs help a ton for SAC
vec_env = make_vec_env(make_env, n_envs=4)

model = SAC(
    "MlpPolicy",
    vec_env,
    device="cuda",                   # use your GPU
    verbose=1,
    tensorboard_log="./tb_ur5e_sac",
)

# evaluate as we go
eval_env = make_env()
eval_cb = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=5, best_model_save_path="./best_sac")

model.learn(total_timesteps=200_000, callback=eval_cb)
model.save("sac_ur5e_lift")

# quick deterministic rollout with rendering so you can watch it
watch_env = make_env()
obs, _ = watch_env.reset()
done, trunc = False, False
step_count = 0
print(f"\n🎬 Running trained model visualization...")
while not (done or trunc) and step_count < 400:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = watch_env.step(action)
    step_count += 1
    if step_count % 50 == 0:
        print(f"Step {step_count}: Reward = {reward:.3f}")

print(f"✅ Visualization complete! Total steps: {step_count}")
watch_env.close()
