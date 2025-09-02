import numpy as np

import os
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from sb3_contrib import TQC

from stable_baselines3.common.vec_env import DummyVecEnv,VecVideoRecorder

from stable_baselines3.common.monitor import Monitor

import torch

# Set seeds for reproducibility
SEED = 42
# random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
eval_env.reset(seed=SEED)




# Base directory for everything
BASE_DIR = "/home/janak/Documents/Pytorch_CPU/Stable_Baseline/Robo/RoboSlide/tqc_roboslide_10M"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Subdirectories

MODEL_DIR = os.path.join(BASE_DIR, "models")


Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
BEST_HER_PATH = os.path.join(MODEL_DIR, "best_model_replay_buffer.pkl")



def load_model(env,device):
    model = None

    # Case 1: Try loading best model + buffer
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_HER_PATH):
        print("Loading BEST model and buffer...")
        model = TQC.load(BEST_MODEL_PATH, env=env, device=device)
        model.load_replay_buffer(BEST_HER_PATH)

    
    else:
        print("No saved model found")
        # model = TQC(**model_kwargs)

    return model



model = load_model(env,device)


# import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

video_folder = os.path.join(BASE_DIR, "videos")
os.makedirs(video_folder, exist_ok=True)
video_length = 1000

eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
eval_env = VecVideoRecorder(
    eval_env,
    video_folder,
    record_video_trigger=lambda step: step == 0,
    video_length=video_length,
    name_prefix=f"tqc-agent-{env_id}"
)

obs= eval_env.reset()

successes = []
episode_success = []

for step in range(video_length):
    action, _ = model.predict(obs, deterministic=True)   # TQC actions
    obs, rewards, dones, infos = eval_env.step(action)

    # Each env in DummyVecEnv returns a list, so we take index [0]
    if "is_success" in infos[0]:
        episode_success.append(infos[0]["is_success"])

    if dones[0]:
        # Store success at the end of episode
        if len(episode_success) > 0:
            successes.append(float(episode_success[-1]))
        episode_success = []
        obs = eval_env.reset()

eval_env.close()

# =========================================================
# Report success rate
# =========================================================
if successes:
    success_rate = sum(successes) / len(successes)
    print(f"Evaluated {len(successes)} episodes")
    print(f"Success Rate: {success_rate*100:.2f}%")
else:
    print("No completed episodes during evaluation")
    
    