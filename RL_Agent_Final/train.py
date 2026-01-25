
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def make_env(rank, seed=0, reward_config=None):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Mask Structure Features (Indices 3, 4, 5, 6) based on Feature Ablation Study
        env = RenkoFilterFastEnv(
            renko_path='Data/Processed/renko_with_predictions.csv',
            states_path='Data/Processed/renko_states.npy',
            reward_config=reward_config,
            mode='train',
            split_ratio=0.8,
            mask_indices=[3, 4, 5, 6]
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    log_dir = "logs/"
    models_dir = "RL_Agent_Final/models"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Configuration
    N_ENVS = 4 # Use 4 cores
    TOTAL_TIMESTEPS = 1000000 # 1M steps for final agent
    
    # Optimized Reward Config from User
    # 12,8.0,-7.2,0.0,67.0,-27.0,-21.0,0.0
    current_reward_config = {
        'win': 8.0,
        'loss': -7.2,
        'be': 0.0,
        'daily_pos_bonus': 67.0,
        'daily_neg_penalty': -27.0,
        'daily_drawdown_penalty': -21.0,
        'step_penalty': 0.0
    }
    
    print(f"Starting Fast Training with {N_ENVS} environments...")
    print(f"Reward Config: {current_reward_config}")
    print("Masking Structure Features: [3, 4, 5, 6]")
    
    # Create Parallel Envs
    env = SubprocVecEnv([make_env(i, reward_config=current_reward_config) for i in range(N_ENVS)])
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=50000 // N_ENVS, save_path=models_dir, name_prefix='final_ppo')
    
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])
    end_time = time.time()
    
    print(f"Training Complete in {(end_time - start_time)/60:.2f} minutes.")
    
    # Save Final Model
    model.save(f"{models_dir}/final_model")
    print("Model Saved.")
    env.close()

if __name__ == "__main__":
    train()
