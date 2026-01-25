
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

try:
    from renko_env_fast import RenkoFilterFastEnv
    from transformer_policy import TransformerExtractor
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv
    from .transformer_policy import TransformerExtractor

def make_env(rank, seed=0, reward_config=None):
    def _init():
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

def train_transformer():
    log_dir = "logs_transformer/"
    models_dir = "RL_Agent_Final/models_transformer"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Config
    N_ENVS = 4
    TOTAL_TIMESTEPS = 1000000 
    N_STACK = 10 # Context Window of 10 bricks
    
    current_reward_config = {
        'win': 8.0,
        'loss': -7.2,
        'be': 0.0,
        'daily_pos_bonus': 67.0,
        'daily_neg_penalty': -27.0,
        'daily_drawdown_penalty': -21.0,
        'step_penalty': 0.0
    }
    
    print(f"Starting Transformer PPO Training...")
    print(f"Stack Size: {N_STACK}")
    
    # 1. Create Base Envs
    env = SubprocVecEnv([make_env(i, reward_config=current_reward_config) for i in range(N_ENVS)])
    
    # 2. Wrap with FrameStack
    # VecFrameStack automatically stacks observations: (Batch, N_Stack, Features) -> Flattened to (Batch, N*F)
    env = VecFrameStack(env, n_stack=N_STACK)
    
    # 3. Initialize PPO with Custom Policy
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(n_stack=N_STACK, d_model=64, n_head=4, n_layers=2),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])] # Smaller MLP head since Transformer does heavy lifting
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        n_steps=2048 // N_ENVS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=50000 // N_ENVS, save_path=models_dir, name_prefix='final_transformer')
    
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])
    end_time = time.time()
    
    print(f"Training Complete in {(end_time - start_time)/60:.2f} minutes.")
    
    model.save(f"{models_dir}/final_model_transformer")
    print("Transformer Model Saved.")
    env.close()

if __name__ == "__main__":
    train_transformer()
