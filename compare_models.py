#!/usr/bin/env python3
"""
Test the best SAC model checkpoint from training
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
import numpy as np

def make_env(render=False):
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    env = TimeLimit(env, max_episode_steps=400)
    return env

def compare_models():
    """Compare the final model vs best checkpoint"""
    print("🔍 Comparing final model vs best checkpoint...")
    
    # Load both models
    final_model = SAC.load("sac_ur5e_lift")
    best_model = SAC.load("best_sac/best_model")
    
    test_env = make_env(render=False)
    
    models = [
        ("Final Model", final_model),
        ("Best Checkpoint", best_model)
    ]
    
    for model_name, model in models:
        print(f"\n📊 Testing {model_name}...")
        episode_rewards = []
        
        for episode in range(3):
            obs, _ = test_env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 400:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
            
            episode_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.3f}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"  Average: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    
    test_env.close()
    
    # Demonstrate the better model with rendering
    print(f"\n🎬 Demonstrating best checkpoint with rendering...")
    render_env = make_env(render=True)
    obs, _ = render_env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    print("Watch the UR5e robot attempt to lift the cube!")
    
    while not done and step_count < 400:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"  Step {step_count}: Total reward = {episode_reward:.3f}")
    
    render_env.close()
    print(f"🎯 Demo complete! Final reward: {episode_reward:.3f}")
    print("✅ Your trained robot is ready for action! 🤖")

if __name__ == "__main__":
    compare_models()
