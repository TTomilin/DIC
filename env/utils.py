import os
from typing import Callable

import gymnasium as gym
import wandb
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv

ALGO_FACTORY = {
    'PPO': PPO,
    'A2C': A2C,
    'DQN': DQN,
    'SAC': SAC,
    'TD3': TD3,
}


class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, velocity_scale=10.0):
        super().__init__(env)
        self.velocity_scale = velocity_scale

    def reward(self, reward):
        velocity = abs(self.env.unwrapped.state[1])
        reward += velocity * self.velocity_scale
        return reward


def make_env(env_name, seed=None):
    env = gym.make(env_name, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    if 'MountainCar' in env_name:
        env = MountainCarRewardWrapper(env)
    return Monitor(env)


def create_model(args, env, log_dir, policy):
    algorithm = ALGO_FACTORY[args.algorithm]
    if args.algorithm == 'A2C':
        model = algorithm(policy, env, verbose=1, tensorboard_log=log_dir, learning_rate=args.learning_rate,
                          gamma=args.gamma, seed=args.seed)
    else:
        model = algorithm(policy, env, verbose=1, tensorboard_log=log_dir, learning_rate=args.learning_rate,
                          gamma=args.gamma, seed=args.seed, batch_size=args.batch_size)
    return model


class MultiSegmentVecVideoRecorder(VecVideoRecorder):
    """
    Extended VecVideoRecorder that updates video_name/video_path
    each time recording starts, avoiding overwriting old files.
    """

    def __init__(
            self,
            venv: VecEnv,
            video_folder: str,
            record_video_trigger: Callable[[int], bool],
            video_length: int = 200,
            name_prefix: str = "rl-video",
    ):
        super().__init__(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=record_video_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
        )

    def _start_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        self.video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}.mp4"
        self.video_path = os.path.join(self.video_folder, self.video_name)
        self.recording = True


class MultiSegmentWandBVecVideoRecorder(MultiSegmentVecVideoRecorder):
    """
    Extended the MultiSegmentVecVideoRecorder to log videos to WandB each time recording is finished.
    """

    def __init__(
            self,
            venv: VecEnv,
            video_folder: str,
            record_video_trigger: Callable[[int], bool],
            video_length: int = 200,
            name_prefix: str = "rl-video",
            log_to_wandb: bool = True,
    ):
        super().__init__(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=record_video_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
        )
        self.log_to_wandb = log_to_wandb

    def _stop_recording(self) -> None:
        """Stop current recording, save video, and log it to wandb."""
        super()._stop_recording()
        if self.log_to_wandb and wandb.run is not None and os.path.exists(self.video_path):
            wandb.log({"video": wandb.Video(self.video_path)})
