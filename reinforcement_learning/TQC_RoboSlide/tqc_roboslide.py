import os
from pathlib import Path
import glob
import gymnasium as gym
import gymnasium_robotics
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv,VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import warnings
warnings.filterwarnings('ignore')


env_id = "FetchSlideDense-v4"

def make_env():
    return Monitor(gym.make(env_id))

env = DummyVecEnv([make_env for _ in range(4)])


# Separate evaluation env (not vectorized, just Monitor)
eval_env = Monitor(gym.make(env_id))


# Base directory for everything
BASE_DIR = "/home/janak/Documents/Pytorch_CPU/Stable_Baseline/Robo/RoboSlide/tqc_roboslide_10M"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Subdirectories
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "robot_slide_sac.zip")
HER_PATH = os.path.join(MODEL_DIR, "her_robot_slide_sac.pkl")
EVAL_LOG_DIR = os.path.join(LOG_DIR, "eval")

Path(EVAL_LOG_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
BEST_HER_PATH = os.path.join(MODEL_DIR, "best_model_replay_buffer.pkl")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("HER_PATH:", HER_PATH)
print("LOG_DIR:", LOG_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("EVAL_LOG_DIR:", EVAL_LOG_DIR)

model_kwargs = dict(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    learning_starts=2000,
    tensorboard_log=LOG_DIR,   # <-- logs go here
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    tau=0.05
)



class EvalCallbackWithHER(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_best_mean_reward = -float("inf")  # track improvements

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Only run check at evaluation steps
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Check if mean reward improved
            if self.last_mean_reward is not None and self.last_mean_reward > self._last_best_mean_reward:
                self._last_best_mean_reward = self.last_mean_reward

                # Save replay buffer alongside the already-saved best model
                model_file = os.path.join(self.best_model_save_path, "best_model.zip")
                if os.path.exists(model_file):
                    buffer_file = model_file.replace(".zip", "_replay_buffer.pkl")
                    self.model.save_replay_buffer(buffer_file)
                    print(f" New best model found, saved HER replay buffer to {buffer_file}")

        return result


eval_callback = EvalCallbackWithHER(
    eval_env,
    n_eval_episodes=10,
    eval_freq=1000,
    log_path=EVAL_LOG_DIR,
    best_model_save_path=MODEL_DIR,  # Best model goes here
    deterministic=True,
    render=False
)



def load_or_create_model(env, model_kwargs, device):
    model = None

    # Case 1: Try loading best model + buffer
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_HER_PATH):
        print("Loading BEST model and buffer...")
        model = TQC.load(BEST_MODEL_PATH, env=env, device=device)
        model.load_replay_buffer(BEST_HER_PATH)
    
    # Case 2: Try loading normal model + buffer
    elif os.path.exists(MODEL_PATH) and os.path.exists(HER_PATH):
        print("Best model not found. Loading latest saved model and buffer...")
        model = TQC.load(MODEL_PATH, env=env, device=device)
        model.load_replay_buffer(HER_PATH)

    # Case 3: Create new model
    else:
        print("No saved model found. Creating a NEW model...")
        model = TQC(**model_kwargs)

    return model



model = load_or_create_model(env, model_kwargs, device)

TOTAL_TIMESTEPS = int(1e7)




model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=True,
    progress_bar=True,
    tb_log_name="her_tqc_run1",
    callback=eval_callback   # <-- add evaluation callback here
)


model.save(MODEL_PATH)
#Save the replay buffer too
model.save_replay_buffer(HER_PATH)
