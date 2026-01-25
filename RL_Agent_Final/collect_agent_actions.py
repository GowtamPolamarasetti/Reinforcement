
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

try:
    from renko_env_fast import RenkoFilterFastEnv
    from transformer_policy import TransformerExtractor
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv
    from .transformer_policy import TransformerExtractor

# Paths
RENKO_PATH = 'Data/Processed/renko_with_predictions_USDMXN.csv'
STATES_PATH = 'Data/Processed/renko_states_USDMXN.npy'
OUTPUT_CSV = 'Data/Processed/agent_actions_usdmxn.csv'

def make_env(stacked=False):
    def _init():
        return RenkoFilterFastEnv(
            renko_path=RENKO_PATH,
            states_path=STATES_PATH,
            reward_config=None,
            mode='test',
            split_ratio=0.0, # Full Data
            mask_indices=[3, 4, 5, 6] 
        )
    if stacked:
        env = DummyVecEnv([_init])
        env = VecFrameStack(env, n_stack=10)
        return env
    else:
        env = DummyVecEnv([_init])
        return env

def collect_actions():
    print("Loading Models...")
    
    # Load Models
    models = {}
    
    # 1. Standard PPO
    try:
        models['ppo'] = PPO.load("RL_Agent_Final/models/final_model")
        print("Loaded PPO")
    except: print("Failed to load PPO")

    # 2. Standard DQN
    try:
        models['dqn'] = DQN.load("RL_Agent_Final/models_dqn/final_model_dqn")
        print("Loaded DQN")
    except: print("Failed to load DQN")
    
    # 3. QR-DQN
    try:
        models['qrdqn'] = QRDQN.load("RL_Agent_Final/models_qrdqn/final_model_qrdqn")
        print("Loaded QR-DQN")
    except: print("Failed to load QR-DQN")
    
    # 4. Recurrent PPO
    try:
        models['recurrent'] = RecurrentPPO.load("RL_Agent_Final/models_recurrent/final_model_recurrent")
        print("Loaded Recurrent PPO")
    except: print("Failed to load Recurrent PPO")
    
    # 5. Transformer PPO
    try:
        models['transformer'] = PPO.load("RL_Agent_Final/models_transformer/final_model_transformer", 
                                         custom_objects={'features_extractor_class': TransformerExtractor})
        print("Loaded Transformer PPO")
    except: print("Failed to load Transformer PPO")
    
    # Envs
    env_std = make_env(stacked=False)
    env_stack = make_env(stacked=True)
    
    base_env = env_std.envs[0]
    num_steps = len(base_env.renko_df)
    print(f"Processing {num_steps} steps...")
    
    # Prepare DataFrame
    # columns: date, outcome, ppo, dqn, qrdqn, recurrent, transformer
    
    # We can just iterate through the dataset for predictions.
    # Note: For Recurrent/Transformer, we must respect sequential order.
    
    # Storage
    actions = {k: np.zeros(num_steps, dtype=int) for k in models.keys()}
    outcomes = []
    
    # Recurrent State
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    obs_std = env_std.reset()
    obs_stack = env_stack.reset()
    
    # Loop
    # We need to run step by step to update Recurrent/Stack states correctly.
    
    # Use tqdm if possible
    from tqdm import tqdm
    
    # We will use base_env.days to iterate? No, we need step-by-step
    # We can just loop until done.
    
    # But wait, RenkoFilterFastEnv in mode='test' with split_ratio=0.0 iterates ALL valid days.
    # But does it skip gaps between days? 
    # The `reset` logic picks a day. `step` iterates bricks in that day.
    # We need to align the collected actions with the dataframe rows.
    # The `current_step_idx` in env tells us the global index in dataframe.
    
    # We should run the simulation and record (global_idx, action).
    # Initialize array with -1 (meaning no trade/skipped/not visited)
    
    global_actions = {k: np.full(num_steps, -1, dtype=int) for k in models.keys()}
    global_outcomes = np.full(num_steps, 'UNKNOWN', dtype=object)
    
    # PnL Outcome Map
    # We can pre-fill outcomes from DF
    df_outcomes = base_env.renko_df['outcome'].values
    
    # Run Simulation
    # We need to run day by day
    
    days = base_env.days
    
    for day in tqdm(days):
        # Set env to this day manually to ensure order
        base_env.current_day_val = day
        env_std.reset() # This normally randomizes, but we hacked it? 
        # Wait, my env reset logic in 'test' mode iterates sequentially:
        # if self.current_day_idx >= len(self.available_days): ...
        # So if we just call reset(), it should be correct sequence if we start fresh.
        pass

    # Actually, simpler: just run the loop until we exhaust all days.
    # Reset envs to start
    base_env.current_day_idx = 0
    obs_std = env_std.reset()
    
    # For stack env, we need to sync it.
    # Stack env wraps a COPY of the env. We need to ensure they are serving the SAME data.
    # This is tricky with DummyVecEnv.
    # A better way: Run 'standard' agents, record. Then Restart, run 'transformer' agent, record.
    # This ensures exact state alignment without fighting DummyVecEnv sync.
    
    print("Collecting Recurrent & Standard Agents...")
    
    # 1. Run Standard + Recurrent Loop
    # They share env_std
    base_env.current_day_idx = 0 # Start from day 0
    obs = env_std.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    while True:
        idx = base_env.current_step_idx
        
        # Predictions
        if 'ppo' in models: 
            act, _ = models['ppo'].predict(obs, deterministic=True)
            global_actions['ppo'][idx] = act
            
        if 'dqn' in models:
            act, _ = models['dqn'].predict(obs, deterministic=True)
            global_actions['dqn'][idx] = act
            
        if 'qrdqn' in models:
            act, _ = models['qrdqn'].predict(obs, deterministic=True)
            global_actions['qrdqn'][idx] = act
            
        if 'recurrent' in models:
            act, lstm_states = models['recurrent'].predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            global_actions['recurrent'][idx] = act
        
        # Step
        # The choice of action for stepping doesn't affect NEXT observation in this env 
        # (State transition is independent of action, except for PnL/Time features, but let's assume PnL=0/neutral path for collection or just use one agent's action)
        # Actually, PnL *is* a feature (index 15).
        # Depending on who "drives", the PnL feature changes.
        # Ideally, for an ensemble, we want to know what they would do given the "Neutral" or "Real" state.
        # But Recurrent PPO might depend on its own PnL history?
        # In our feature list, PnL is the DAILY PnL.
        # If we just step with Action=0 (Skip), PnL stays 0.
        # This provides a consistent "Baseline" view for all agents. 
        # If we let PPO drive, PnL reflects PPO.
        # Let's step with Action 0 (SKIP) to keep PnL at 0 (Clean State).
        # This tests "Independent Decision Making" without PnL momentum bias.
        
        obs, rewards, dones, infos = env_std.step(np.array([0])) 
        episode_starts = dones
        
        # Check if we looped back to start (test mode loop)
        # Env logic: reset() increments idx. 
        # Monitoring `base_env.current_day_idx`. 
        # If we finish all days, we should stop.
        # But VecEnv keeps resetting.
        
        if dones[0]:
            # Check if we finished all days
            if base_env.current_day_idx >= len(base_env.days):
                break
                
    # 2. Run Transformer (Stacked Env)
    print("Collecting Transformer Agent...")
    env_stack_base = env_stack.envs[0]
    env_stack_base.current_day_idx = 0
    obs = env_stack.reset()
    
    while True:
        idx = env_stack_base.current_step_idx
        
        if 'transformer' in models:
            act, _ = models['transformer'].predict(obs, deterministic=True)
            global_actions['transformer'][idx] = act
            
        obs, rewards, dones, infos = env_stack.step(np.array([0]))
        
        if dones[0]:
            if env_stack_base.current_day_idx >= len(env_stack_base.days):
                break

    # Save
    df = pd.DataFrame(global_actions)
    df['outcome'] = df_outcomes
    
    # Create Date column
    df['date'] = base_env.renko_df['date']
    
    # Filter rows that were never visited (should be none if we ran full days)
    # But step 0 of a day is visited.
    # Check -1 values
    valid_rows = df['ppo'] != -1
    df = df[valid_rows]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved actions to {OUTPUT_CSV}. Shape: {df.shape}")

if __name__ == "__main__":
    collect_actions()
