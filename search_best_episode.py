#!/usr/bin/env python3
"""
Search extensively for the highest reward episodes and create GIF
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
import numpy as np
import json
import pickle
import time

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

def extensive_search_for_best():
    """Search extensively for the absolute best episode"""
    print("🔍 EXTENSIVE SEARCH FOR 20+ REWARD EPISODES")
    print("="*50)
    
    best_model = SAC.load("best_sac/best_model")
    env = make_env(render=False)
    
    all_episodes = []
    best_reward = -float('inf')
    best_episode_data = None
    
    # Try 200 episodes with different seeds
    for episode in range(200):
        # Use different random seeds
        seed = episode * 17 + 42  # Different seed pattern
        np.random.seed(seed)
        env.action_space.seed(seed)
        
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        episode_actions = []
        
        while not done and step_count < 400:
            action, _ = best_model.predict(obs, deterministic=True)
            episode_actions.append(action.copy())
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        episode_data = {
            'episode': episode + 1,
            'seed': seed,
            'reward': episode_reward,
            'steps': step_count,
            'actions': episode_actions
        }
        
        all_episodes.append(episode_data)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode_data = episode_data.copy()
            print(f"🏆 NEW BEST! Episode {episode + 1} (seed {seed}): {episode_reward:.3f}")
            
            # If we find a really high reward, celebrate
            if episode_reward > 20.0:
                print(f"⭐ JACKPOT! Found 20+ reward episode!")
                break
        
        # Progress indicator
        if (episode + 1) % 25 == 0:
            top_rewards = sorted([ep['reward'] for ep in all_episodes], reverse=True)[:5]
            print(f"📊 Progress: {episode + 1}/200 episodes. Top 5 rewards: {[round(r, 2) for r in top_rewards]}")
    
    env.close()
    
    # Save all episode data
    with open('all_episodes_data.pkl', 'wb') as f:
        pickle.dump(all_episodes, f)
    
    # Sort and show top episodes
    all_episodes.sort(key=lambda x: x['reward'], reverse=True)
    
    print(f"\n🎯 SEARCH COMPLETE!")
    print(f"📊 Top 10 Episodes:")
    for i, ep in enumerate(all_episodes[:10]):
        print(f"   {i+1}. Episode {ep['episode']} (seed {ep['seed']}): {ep['reward']:.3f}")
    
    # Save the best episode
    with open('absolute_best_episode.pkl', 'wb') as f:
        pickle.dump(best_episode_data, f)
    
    print(f"\n🏆 ABSOLUTE BEST:")
    print(f"   Episode: {best_episode_data['episode']}")
    print(f"   Seed: {best_episode_data['seed']}")
    print(f"   Reward: {best_episode_data['reward']:.3f}")
    print(f"   Actions saved: {len(best_episode_data['actions'])}")
    
    return best_episode_data

def replay_absolute_best():
    """Replay the absolute best episode found"""
    try:
        with open('absolute_best_episode.pkl', 'rb') as f:
            episode_data = pickle.load(f)
    except FileNotFoundError:
        print("❌ No absolute best episode found. Run search first.")
        return
    
    print(f"\n🎬 REPLAYING ABSOLUTE BEST EPISODE")
    print(f"   Reward: {episode_data['reward']:.3f}")
    print(f"   Seed: {episode_data['seed']}")
    print("="*50)
    
    # Set the exact same seed
    np.random.seed(episode_data['seed'])
    
    env = make_env(render=True)
    env.action_space.seed(episode_data['seed'])
    
    obs, _ = env.reset()
    episode_reward = 0
    
    print("🤖 Replaying the absolute best performance...")
    print("📹 This is the perfect episode to record!")
    
    for step, action in enumerate(episode_data['actions']):
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Show milestone progress
        if step % 50 == 0:
            print(f"  Step {step}: Reward = {episode_reward:.3f}")
        
        if done:
            break
    
    env.close()
    
    print(f"\n🎯 Replay Results:")
    print(f"   Original reward: {episode_data['reward']:.3f}")
    print(f"   Replayed reward: {episode_reward:.3f}")
    print(f"   Match quality: {'✅ Perfect' if abs(episode_reward - episode_data['reward']) < 0.1 else '⚠️ Close'}")

def manual_recording_session():
    """Guide for manual recording"""
    print("\n🎬 MANUAL RECORDING SESSION")
    print("="*40)
    
    try:
        with open('absolute_best_episode.pkl', 'rb') as f:
            episode_data = pickle.load(f)
    except FileNotFoundError:
        print("❌ No episode data found. Run search first.")
        return
    
    print(f"🎯 About to replay episode with {episode_data['reward']:.3f} reward!")
    print("\n📱 Recording Setup:")
    print("   1. 🎥 Start your screen recording (Win+G, OBS, or phone)")
    print("   2. 📍 Make sure MuJoCo window is visible")
    print("   3. ⏰ Recording will start in 5 seconds")
    print("\n🎞️ For GIF conversion:")
    print("   • Use online converter: https://ezgif.com/video-to-gif")
    print("   • Settings: 320x320, 10-15 FPS, loop enabled")
    
    input("\n📹 Press Enter when your recording is ready...")
    
    print("\n🚀 Starting in:")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🎬 RECORDING NOW!")
    replay_absolute_best()
    print("\n🎬 RECORDING COMPLETE!")

def show_statistics():
    """Show statistics from the search"""
    try:
        with open('all_episodes_data.pkl', 'rb') as f:
            all_episodes = pickle.load(f)
    except FileNotFoundError:
        print("❌ No episode data found. Run search first.")
        return
    
    rewards = [ep['reward'] for ep in all_episodes]
    rewards.sort(reverse=True)
    
    print(f"\n📊 EPISODE STATISTICS:")
    print(f"   Total episodes tested: {len(all_episodes)}")
    print(f"   Highest reward: {max(rewards):.3f}")
    print(f"   Average reward: {np.mean(rewards):.3f}")
    print(f"   Episodes with 10+ reward: {sum(1 for r in rewards if r >= 10.0)}")
    print(f"   Episodes with 15+ reward: {sum(1 for r in rewards if r >= 15.0)}")
    print(f"   Episodes with 20+ reward: {sum(1 for r in rewards if r >= 20.0)}")
    
    # Show top 15 episodes
    print(f"\n🏆 TOP 15 EPISODES:")
    all_episodes.sort(key=lambda x: x['reward'], reverse=True)
    for i, ep in enumerate(all_episodes[:15]):
        print(f"   {i+1:2d}. Episode {ep['episode']:3d} (seed {ep['seed']:4d}): {ep['reward']:6.3f}")

def main():
    """Main menu"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "search":
            extensive_search_for_best()
        elif command == "replay":
            replay_absolute_best()
        elif command == "record":
            manual_recording_session()
        elif command == "stats":
            show_statistics()
        else:
            print("Usage: python search_best_episode.py [search|replay|record|stats]")
    else:
        print("🤖 ROBOT EPISODE ANALYZER")
        print("Available commands:")
        print("  python search_best_episode.py search  # Find the absolute best episodes")
        print("  python search_best_episode.py replay  # Replay the best episode")
        print("  python search_best_episode.py record  # Guided recording session")
        print("  python search_best_episode.py stats   # Show episode statistics")

if __name__ == "__main__":
    main()
