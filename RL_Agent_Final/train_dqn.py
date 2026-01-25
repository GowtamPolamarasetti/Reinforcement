
import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
# DQN in SB3 defaults to ReplayBuffer. 

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def make_env(rank, seed=0, reward_config=None):
    """
    Utility function for env.
    """
    def _init():
        # Mask Structure Features (Indices 3, 4, 5, 6)
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
    models_dir = "RL_Agent_Final/models_dqn"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Configuration
    # DQN is off-policy. Parallel envs increase diversity in replay buffer.
    N_ENVS = 4 
    TOTAL_TIMESTEPS = 1000000 
    
    # Same Optimized Reward Config as PPO
    current_reward_config = {
        'win': 8.0,
        'loss': -7.2,
        'be': 0.0,
        'daily_pos_bonus': 67.0,
        'daily_neg_penalty': -27.0,
        'daily_drawdown_penalty': -21.0,
        'step_penalty': 0.0
    }
    
    print(f"Starting DQN Training with {N_ENVS} environments...")
    
    # Create Envs
    env = SubprocVecEnv([make_env(i, reward_config=current_reward_config) for i in range(N_ENVS)])
    
    # Initialize DQN
    # MlpPolicy is standard.
    # exploration_fraction: Decay epsilon over first 10% of training.
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=1e-4, # Slightly lower LH for stability
        buffer_size=100000, 
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=50000 // N_ENVS, save_path=models_dir, name_prefix='final_dqn')
    
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])
    end_time = time.time()
    
    print(f"Training Complete in {(end_time - start_time)/60:.2f} minutes.")
    
    # Save Final Model
    model.save(f"{models_dir}/final_model_dqn")
    print("Model Saved.")
    env.close()

if __name__ == "__main__":
    train()
