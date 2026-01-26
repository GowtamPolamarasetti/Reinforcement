import os

# Project Root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subdirectories
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
EXECUTION_DIR = os.path.join(ROOT_DIR, 'execution')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Create Logs dir if not exists
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Model Paths (Relative to Trader/models or absolute)
# As per user's provided structure, models are in Trader/Models (capital M) or copied there.
# We will standardize on using the directory `Trader/models` (lowercase) or mapping to the existing `Models` dir.
# Let's use the capitalized `Models` dir since that's what exists, or we rename it.
# The `implementation_plan` suggested `models` (lowercase). 
# I will accept `Models` (capitalized) as the source of truth if it exists, or `models`.
# For now, let's point to the existing `Models` directory which is a sibling to `config`? 
# Wait, the user said "Everything... is in @[Models]".
# `Trader/Models` exists (I listed it).
MODELS_ROOT = os.path.join(ROOT_DIR, 'Models') 

# Model Specific Paths
M_PPO_PATH = os.path.join(MODELS_ROOT, 'models_ppo', 'final_model')
M_DQN_PATH = os.path.join(MODELS_ROOT, 'models_dqn', 'final_model_dqn')
M_QRDQN_PATH = os.path.join(MODELS_ROOT, 'models_qrdqn', 'final_model_qrdqn')
M_RECURRENT_PATH = os.path.join(MODELS_ROOT, 'models_recurrent', 'final_model_recurrent')
M_TRANSFORMER_PATH = os.path.join(MODELS_ROOT, 'models_transformer', 'final_model_transformer')

# Outcome Models (CatBoost/HMM)
# Checking directory structure from `list_dir` output of `Trader/Models`:
# {"name":"Outcome","isDir":true,"numChildren":5}
# So they are in `Trader/Models/Outcome`
OUTCOME_DIR = os.path.join(MODELS_ROOT, 'Outcome')
M_CATBOOST_BINARY = os.path.join(OUTCOME_DIR, 'binary_model.pkl')
M_CATBOOST_MULTI = os.path.join(OUTCOME_DIR, 'stacked_model.pkl') # Assuming this is the multi or stacked? 
# Based on file listing from previous turn (Step 12):
# binary_model.pkl, outcome_predictor.pkl, stacked_model.pkl.
# We need to map these correctly in settings.

