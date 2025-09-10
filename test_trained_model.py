#!/usr/bin/env python3
"""
Test the trained SAC model on the Lift task
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
        has_renderer=render,          # Enable rendering to watch the robot
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    env = TimeLimit(env, max_episode_steps=400)
    return env

def test_trained_model():
    """Test the trained SAC model"""
    print("🤖 Loading trained SAC model...")
    
    try:
        model = SAC.load("sac_ur5e_lift")
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file 'sac_ur5e_lift.zip' not found!")
        print("Run the training script (lift_SAC.py) first to train a model.")
        return
    
    # Test with rendering disabled first (faster)
    print("\n📊 Testing model performance (no rendering)...")
    test_env = make_env(render=False)
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
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
        episode_lengths.append(step_count)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Steps = {step_count}")
    
    test_env.close()
    
    # Print statistics
    print(f"\n📈 Results over {num_episodes} episodes:")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best episode reward: {max(episode_rewards):.3f}")
    
    # Test with rendering (you can watch the robot)
    print(f"\n🎬 Running one episode with rendering...")
    print("Note: This will open a MuJoCo viewer window")
    
    render_env = make_env(render=True)
    obs, _ = render_env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    while not done and step_count < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"  Step {step_count}: Reward = {episode_reward:.3f}")
    
    render_env.close()
    print(f"🎯 Final episode: Reward = {episode_reward:.3f}, Steps = {step_count}")
    print("✅ Model testing complete!")

if __name__ == "__main__":
    test_trained_model()
