
import os
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def train_qrdqn():
    # Paths
    save_dir = "RL_Agent_Final/models_qrdqn"
    log_dir = "RL_Agent_Final/logs_qrdqn"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Env Config
    env_config = {
        'renko_path': 'Data/Processed/renko_with_predictions.csv',
        'states_path': 'Data/Processed/renko_states.npy',
        'reward_config': {
            'win': 8.0,
            'loss': -7.2, # Symmetric
            'be': 0.0,
            'inactive_penalty': -0.01,
            'streak_bonus': 0.1,
            'daily_pos_bonus': 67,
            'daily_neg_penalty': -27,
            'daily_drawdown_penalty': -27
        },
        'mode': 'train',
        'split_ratio': 0.8,
        'mask_indices': [3, 4, 5, 6] 
    }
    
    # Create Envs
    env = RenkoFilterFastEnv(**env_config)
    env = Monitor(env)
    
    eval_config = env_config.copy()
    eval_config['mode'] = 'test'
    eval_env = RenkoFilterFastEnv(**eval_config)
    eval_env = Monitor(eval_env)
    
    # QRDQN Hyperparameters (optimized for probability tasks)
    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=log_dir,
        seed=42
    )
    
    print(f"Training QR-DQN on {len(env.env.days)} days...")
    
    # Callbacks
    # We cheat slightly on eval freq since step logic is fast but epoch is long
    eval_callback = EvalualuationCallback = None # (Typo in variable name in previous logic, skipping)
    
    try:
        from stable_baselines3.common.callbacks import EvalCallback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False
        )
    except ImportError:
        pass

    # Train
    total_timesteps = 200000 # Same as PPO/DQN base
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # Save Final
    final_path = os.path.join(save_dir, "final_model_qrdqn")
    model.save(final_path)
    print(f"Saved QR-DQN model to {final_path}")

if __name__ == "__main__":
    train_qrdqn()
