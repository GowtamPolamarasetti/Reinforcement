from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, RecurrentPPO
import numpy as np
from config.definitions import (
    M_PPO_PATH, M_DQN_PATH, M_QRDQN_PATH, M_RECURRENT_PATH, M_TRANSFORMER_PATH
)
from config.settings import WEIGHTS, VOTE_THRESHOLD
from utils.logger import logger
import os

# Try to import Transformer Policy
# We need `transformer_policy.py` in the path or same dir.
# It is in `RL_Agent_Final/transformer_policy.py`.
# We should copy it to `Trader/models` or `Trader/utils` to import it.
import sys
# sys.path.append(...) or copy.
# Let's assume we copy it to `Trader/models/transformer_policy.py` later.
# For now, import inside try/except block.

class EnsembleAgent:
    def __init__(self):
        self.models = {}
        
    def load_all(self):
        logger.info("Loading Ensemble Agents...")
        
        # Helper to load safely
        def load_one(name, cls, path, **kwargs):
            if os.path.exists(path + ".zip") or os.path.exists(path):
                try:
                    self.models[name] = cls.load(path, **kwargs)
                    logger.info(f"Loaded {name}")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Model {name} not found at {path}")

        load_one('ppo', PPO, M_PPO_PATH)
        load_one('dqn', DQN, M_DQN_PATH)
        load_one('qrdqn', QRDQN, M_QRDQN_PATH)
        load_one('recurrent', RecurrentPPO, M_RECURRENT_PATH)
        
        # Transformer needs custom object
        # PICKLE FIX: Map 'RL_Agent_Final.algo.transformer_policy' to 'models.transformer_policy'
        try:
            import sys
            import types
            from models import transformer_policy
            
            # Create dummy 'RL_Agent_Final' module
            if 'RL_Agent_Final' not in sys.modules:
                rl_agent_final = types.ModuleType('RL_Agent_Final')
                sys.modules['RL_Agent_Final'] = rl_agent_final
                
                # Create dummy 'RL_Agent_Final.algo'
                algo = types.ModuleType('RL_Agent_Final.algo')
                rl_agent_final.algo = algo
                sys.modules['RL_Agent_Final.algo'] = algo
                
                # Create dummy 'RL_Agent_Final.transformer_policy' (Attempt 1)
                rl_agent_final.transformer_policy = transformer_policy
                sys.modules['RL_Agent_Final.transformer_policy'] = transformer_policy
                
                # Create dummy 'RL_Agent_Final.algo.transformer_policy' (Attempt 2 - nested)
                algo.transformer_policy = transformer_policy
                sys.modules['RL_Agent_Final.algo.transformer_policy'] = transformer_policy
                
            from models.transformer_policy import TransformerExtractor
            load_one('transformer', PPO, M_TRANSFORMER_PATH, custom_objects={'features_extractor_class': TransformerExtractor})
            
        except ImportError as e:
            logger.error(f"Transformer Policy module load error: {e}")
        except Exception as e:
            logger.error(f"Transformer Pickle Patched Load Failed: {e}")

    def predict(self, obs, lstm_states=None, episode_starts=None, obs_stack=None):
        """
        Returns: 
            final_action (int): 1 (Buy/Trade) or 0 (Skip)
            new_lstm_states: Updated states for recurrent model
            score: The raw ensemble score
        """
        score = 0.0
        new_lstm_states = lstm_states
        
        # Voting Logic
        # 1. PPO
        if 'ppo' in self.models:
            act, _ = self.models['ppo'].predict(obs, deterministic=True)
            score += (1 if act==1 else -1) * WEIGHTS['ppo']
            
        # 2. DQN
        if 'dqn' in self.models:
            act, _ = self.models['dqn'].predict(obs, deterministic=True)
            score += (1 if act==1 else -1) * WEIGHTS['dqn']
            
        # 3. QRDQN
        if 'qrdqn' in self.models:
            act, _ = self.models['qrdqn'].predict(obs, deterministic=True)
            score += (1 if act==1 else -1) * WEIGHTS['qrdqn']
            
        # 4. Recurrent
        if 'recurrent' in self.models:
            # Recurrent expects (n_envs, obs_dim) usually. DummyVecEnv wraps it.
            # If obs is 1D (21,), reshape to (1, 21)
            obs_reshaped = obs.reshape(1, -1)
            act, new_lstm_states = self.models['recurrent'].predict(
                obs_reshaped, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            score += (1 if act[0]==1 else -1) * WEIGHTS['recurrent']
            
        # 5. Transformer
        if 'transformer' in self.models and obs_stack is not None:
            # Transformer expects Stacked Frames.
            # Shape for vectorized env: (n_envs, n_stack, obs_dim) or (n_envs, n_stack*obs_dim) depending on policy.
            # Our TransformerPolicy expects (Batch, Seq, Feat) = (1, 10, 21).
            # obs_stack is (10, 21). Reshape to (1, 10, 21)?
            # SB3 VecFrameStack flattens observation usually? NO.
            # Only if channel_last=False?
            # Our training used VecFrameStack. The observation space became (10, 21).
            # predict() expects simple numpy array matching observation space.
            # If observation space is Box(10, 21), we pass (10, 21).
            # predictor expects flattened input matching the observation space (Box(210,)).
            # TransformerExtractor inside the policy will reshape it back to (Batch, 10, 21).
            # obs_stack is (10, 21). Flatten to (210,).
            flat_stack = obs_stack.flatten()
            act, _ = self.models['transformer'].predict(flat_stack, deterministic=True)
            score += (1 if act==1 else -1) * WEIGHTS['transformer']
            
        final_action = 1 if score > VOTE_THRESHOLD else 0
        return final_action, new_lstm_states, score
