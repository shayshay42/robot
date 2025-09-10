#!/usr/bin/env python3
"""
Improved SAC training with OSC_POSE controller and vectorized environments
Based on the robomimic + robosuite best practices
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import h5py

def make_env():
    """Create environment with improved settings"""
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,        # Low-dimensional observations
        control_freq=20,
        horizon=300,                 # Shorter horizon as recommended
        reward_shaping=True,         # Dense rewards for faster learning
        hard_reset=False,            # Faster resets
    )
    
    env = GymWrapper(env)  # Flatten observations to 1D
    env = TimeLimit(env, max_episode_steps=300)
    env = Monitor(env)     # For logging
    return env

def prefill_from_hdf5(model, h5_path, max_trajs=100):
    """
    Prefill SAC's replay buffer from robomimic HDF5 rollouts
    This is where we'd load the demonstration data
    """
    if not os.path.exists(h5_path):
        print(f"⚠️ No HDF5 file found at {h5_path}")
        print("   Skipping prefill - will train from scratch")
        return
    
    print(f"📁 Loading demonstration data from {h5_path}")
    
    try:
        with h5py.File(h5_path, "r") as f:
            eps_keys = sorted(f["data"].keys())[:max_trajs]
            total_transitions = 0
            
            for k in eps_keys:
                g = f["data"][k]
                obs = np.array(g["observations"])     # shape [T, obs_dim]
                act = np.array(g["actions"])          # shape [T, act_dim]
                rew = np.array(g["rewards"]).reshape(-1)
                
                # Create done flags (True only at episode end)
                dones = np.zeros(len(rew), dtype=bool)
                dones[-1] = True
                
                # Add transitions to replay buffer
                for t in range(len(rew)-1):
                    model.replay_buffer.add(
                        obs[t], act[t], rew[t], obs[t+1], dones[t],
                        infos={}
                    )
                    total_transitions += 1
            
            print(f"✅ Loaded {total_transitions} transitions from {len(eps_keys)} episodes")
            
    except Exception as e:
        print(f"❌ Error loading HDF5: {e}")
        print("   Continuing with empty replay buffer")

def train_improved_sac():
    """Train SAC with improved setup based on robomimic best practices"""
    print("🚀 Starting Improved SAC Training")
    print("="*50)
    
    # Hyperparameters based on robosuite recommendations
    N_ENVS = 8  # More parallel environments
    TOTAL_TIMESTEPS = 150_000  # Fewer timesteps needed with good setup
    
    print(f"📊 Configuration:")
    print(f"   Environments: {N_ENVS}")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Controller: OSC_POSE (6-DoF Cartesian + gripper)")
    print(f"   Robot: UR5e")
    print(f"   Horizon: 300 steps")
    
    # Create vectorized environments with normalization
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, 
                          norm_obs=True, 
                          norm_reward=True, 
                          clip_obs=10.0,
                          gamma=0.99)
    
    # SAC model with optimized hyperparameters
    model = SAC(
        "MlpPolicy",
        vec_env,
        device="cuda",
        batch_size=512,           # Larger batch size
        learning_rate=3e-4,       # Standard learning rate
        tau=0.01,                 # Soft update rate
        gamma=0.99,               # Discount factor
        train_freq=64,            # Train every 64 steps
        gradient_steps=64,        # Multiple gradient steps
        buffer_size=1_000_000,    # Large replay buffer
        learning_starts=0,        # Start learning immediately (good for prefilled buffer)
        ent_coef="auto",          # Automatic entropy tuning
        verbose=1,
        tensorboard_log="./tb_improved_sac"
    )
    
    # Try to prefill replay buffer with demonstrations (if available)
    # This is where you'd use the robomimic-generated rollouts
    prefill_from_hdf5(model, "lift_bc_seed.hdf5", max_trajs=100)
    
    # Evaluation callback
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        best_model_save_path="./best_improved_sac",
        deterministic=True,
        verbose=1
    )
    
    # Train the model
    print("\n🎯 Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("sac_improved_ur5e_lift")
    vec_env.save("vec_normalize_improved.pkl")
    
    print("✅ Training complete!")
    return model, vec_env

def test_improved_model():
    """Test the improved trained model"""
    print("\n🎬 Testing improved model...")
    
    # Load the trained model
    try:
        model = SAC.load("sac_improved_ur5e_lift")
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ No trained model found. Run training first.")
        return
    
    # Create test environment
    test_env = make_env()
    
    # Run evaluation episodes
    n_episodes = 5
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.3f}, Steps = {step_count}")
    
    test_env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\n📊 Test Results:")
    print(f"   Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"   Best episode: {max(episode_rewards):.3f}")
    print(f"   All episodes: {[round(r, 3) for r in episode_rewards]}")

def compare_with_original():
    """Compare improved model with original"""
    print("\n🔍 Comparing models...")
    
    models = []
    
    # Load original model
    try:
        original_model = SAC.load("sac_ur5e_lift")
        models.append(("Original SAC", original_model))
    except FileNotFoundError:
        print("⚠️ Original model not found")
    
    # Load improved model
    try:
        improved_model = SAC.load("sac_improved_ur5e_lift")
        models.append(("Improved SAC", improved_model))
    except FileNotFoundError:
        print("⚠️ Improved model not found")
    
    if not models:
        print("❌ No models to compare")
        return
    
    test_env = make_env()
    
    for model_name, model in models:
        print(f"\n📊 Testing {model_name}...")
        episode_rewards = []
        
        for episode in range(3):
            obs, _ = test_env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 300:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
            
            episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards)
        print(f"   Average: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    
    test_env.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "train":
            train_improved_sac()
        elif command == "test":
            test_improved_model()
        elif command == "compare":
            compare_with_original()
        else:
            print("Usage: python improved_sac_training.py [train|test|compare]")
    else:
        print("🤖 IMPROVED SAC TRAINING")
        print("Available commands:")
        print("  python improved_sac_training.py train    # Train with improved setup")
        print("  python improved_sac_training.py test     # Test trained model")
        print("  python improved_sac_training.py compare  # Compare with original")
        print("\n🎯 Improvements:")
        print("  • OSC_POSE controller (6-DoF Cartesian control)")
        print("  • VecNormalize for observation/reward normalization")
        print("  • Optimized SAC hyperparameters")
        print("  • Shorter horizon (300 steps)")
        print("  • Support for demonstration prefilling")
        print("  • Better evaluation and monitoring")
