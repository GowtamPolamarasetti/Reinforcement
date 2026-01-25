
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import os

class RenkoFilterFastEnv(gym.Env):
    """
    Super-Optimized Environment that uses pre-computed states.
    Skips all feature engineering at runtime.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, renko_path, states_path, reward_config=None, mode='train', split_ratio=0.8, mask_indices=None):
        super(RenkoFilterFastEnv, self).__init__()
        
        self.mode = mode 
        self.mask_indices = mask_indices 
        
        # Reward Configuration
        self.reward_config = reward_config if reward_config else {
            'win': 1.0,
            'loss': -1.0,
            'be': 0.0,
            'daily_pos_bonus': 20.0,
            'daily_neg_penalty': -20.0,
            'daily_drawdown_penalty': -20.0,
            'step_penalty': 0.0
        }
        
        # Load Data
        self.renko_df = pd.read_csv(renko_path)
        # Dates needed for day splitting
        self.renko_df['date'] = pd.to_datetime(self.renko_df['date'])
        if self.renko_df['date'].dt.tz is not None:
            self.renko_df['date'] = self.renko_df['date'].dt.tz_localize(None)
            
        # Load States
        print(f"Loading pre-computed states from {states_path}...")
        self.states = np.load(states_path)
        
        if len(self.states) != len(self.renko_df):
            raise ValueError(f"State mismatch! CSV has {len(self.renko_df)} rows, States has {len(self.states)} rows.")
            
        # Organize Data by Day
        # We need fast access to bricks for a given day
        self.renko_df['date_only'] = self.renko_df['date'].dt.date
        self.unique_days = self.renko_df['date_only'].unique()
        
        # Split Train/Test
        split_idx = int(len(self.unique_days) * split_ratio)
        if mode == 'train':
            self.days = self.unique_days[:split_idx]
        else:
            self.days = self.unique_days[split_idx:]
            
        # To speed up reset(), let's map day -> (start_idx, length)
        # This prevents filtering dataframe on every reset
        self.day_map = {}
        
        # Group indices
        # This gives us a dict of day -> list of indices
        # But we need contiguous slices if possible. Renko is time-sorted, so they should be contiguous.
        # Let's find start/end for each day.
        
        current_idx = 0
        N = len(self.renko_df)
        
        # We can iterate unique days and find bounds
        # Or faster: groupby
        grouped = self.renko_df.groupby('date_only')
        for day, group in grouped:
            # Check if this day is in our split
            if day in self.days:
                # We need global indices.
                # Assuming df is sorted (which it is), we can just find min/max idx
                indices = group.index.values
                start = indices[0]
                end = indices[-1] + 1 # Exclusive
                self.day_map[day] = (start, end)
                
        self.available_days = list(self.day_map.keys())
        
        # Define Action Space
        self.action_space = spaces.Discrete(2)
        
        # Define Observation Space
        # Shape is (21,)
        self.obs_dim = 21
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # State variables
        self.current_day_val = None
        self.current_step_idx = 0 # Global index
        self.day_end_idx = 0      # Global index
        self.daily_pnl = 0.0
        self.max_daily_loss = -3.0
        self.current_day_idx = 0 # For test sequence

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a day
        if self.mode == 'train':
             self.current_day_val = self.available_days[np.random.randint(len(self.available_days))]
        else:
             if self.current_day_idx >= len(self.available_days):
                 self.current_day_idx = 0
             self.current_day_val = self.available_days[self.current_day_idx]
             self.current_day_idx += 1
             
        # Get bounds
        start, end = self.day_map[self.current_day_val]
        
        self.current_step_idx = start
        self.day_end_idx = end
        
        self.daily_pnl = 0.0
        
        return self._get_observation(), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        # Get outcome from CSV (fast lookup)
        # Accessing underlying numpy array is faster than iloc
        # But we need 'outcome' column. Let's cache outcomes/prices if needed?
        # self.renko_df['outcome'] is fast access if we converted to Categorical or Int
        # For now, iloc is okay-ish but slower than array access.
        # Optimizing: self.outcomes_arr = df['outcome'].values
        # Done in __init__? Let's assume standard iloc for safety first, but we can access column array
        
        outcome = self.renko_df.iat[self.current_step_idx, 8] # Column 8 is outcome? Need to double check index
        # Safer: use pre-loaded numpy arrays for critical path
        
        # Let's verify column index at runtime or by name
        outcome = self.renko_df.iloc[self.current_step_idx]['outcome']
        
        # Execute Action
        if action == 1: # TAKE
            if outcome == 'WIN':
                reward = self.reward_config['win']
                self.daily_pnl += 0.5
            elif outcome == 'LOSS':
                reward = self.reward_config['loss']
                self.daily_pnl -= 0.5
            elif outcome == 'BE' or outcome == 'BREAKEVEN':
                reward = self.reward_config['be'] 
            else:
                 reward = 0.0
        else: # SKIP
            reward = 0.0
            
        # Check Daily Drawdown
        if self.daily_pnl <= self.max_daily_loss:
            terminated = True
            info['reason'] = 'Drawdown Limit'
            reward += self.reward_config['daily_drawdown_penalty']
            
        # Move to next step
        self.current_step_idx += 1
        
        if self.current_step_idx >= self.day_end_idx:
            terminated = True
            info['reason'] = 'End of Day'
            
        # Terminal Rewards at End of Day
        if terminated and info.get('reason') == 'End of Day':
            if self.daily_pnl < 0:
                reward += self.reward_config['daily_neg_penalty']
            elif self.daily_pnl > 0:
                reward += self.reward_config['daily_pos_bonus']
            else:
                reward += 0.0 
            
        # Step Penalty
        reward += self.reward_config.get('step_penalty', 0.0)

        # Get next observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        # 1. Retrieve Pre-computed state
        # Use copy to avoid modifying the source array when we insert dynamic PnL
        obs = self.states[self.current_step_idx].copy()
        
        # 1.5 Apply Feature Mask (if any)
        if self.mask_indices:
            obs[self.mask_indices] = 0.0
        
        # 2. Inject Dynamic PnL
        # PnL is at index 15
        # [Regime(2), LSTM(1), Struct(4), Ind(8), PnL(1), Time(1), Preds(4)]
        obs[15] = np.clip(self.daily_pnl, -5, 5)
        
        # Time Left is already pre-computed in states (index 16)
        
        return obs

    def render(self, mode='human'):
        print(f"Day: {self.current_day_val} Step: {self.current_step_idx} PnL: {self.daily_pnl}")
