import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# other imports...
import mujoco
print(f"MuJoCo version: {mujoco.__version__}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

log_dir = "/home/janak/Documents/Pytorch_CPU/Stable_Baseline/ppo_humanoid_standup_log_20_Million/"
model_path = os.path.join(log_dir, "ppo_humanoid_standup.zip")
os.makedirs(log_dir, exist_ok=True)

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, "ppo_humanoid_standup_latest.zip")
            self.model.save(save_file)
            env.save(os.path.join(self.save_path, "vecnormalize_latest.pkl"))
            if self.verbose > 0:
                print(f"Checkpoint saved at step {self.num_timesteps} -> {save_file}")
        return True

if __name__ == "__main__":
    n_envs = 8
    env = VecNormalize(
        make_vec_env("HumanoidStandup-v4", n_envs=n_envs, vec_env_cls=SubprocVecEnv),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.
    )

    eval_env = VecNormalize(
        make_vec_env("HumanoidStandup-v4", n_envs=1),
        training=False,
        norm_obs=True,
        norm_reward=False
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=log_dir)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    if os.path.exists(model_path):
        print("Loading existing model...")
        model = PPO.load(model_path, env=env, device=device)
        env = model.get_env()
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
        )

    print("Starting training...")
    model.learn(
        total_timesteps=20_000_000,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="ppo_humanoid_standup",
        progress_bar=True
    )

    model.save(model_path)
    env.save(os.path.join(log_dir, "vecnormalize_final.pkl"))
    print("Training completed.")
