import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from env.utils.utils import MultiSegmentWandBVecVideoRecorder, make_env, create_model


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f'{args.algorithm}_{args.env_name}_seed_{args.seed}_{timestamp}'
    base_path = Path(__file__).parent.parent.resolve()
    log_dir = f"{base_path}/logs/{run_name}"
    wandb.init(project=args.project, mode="online", sync_tensorboard=True, name=run_name, dir=base_path, id=run_name,
               save_code=True, group=args.env_name, job_type=args.algorithm, tags=args.wandb_tags, config=vars(args))
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(lambda: make_env(args.env_name, seed=args.seed), n_envs=args.n_envs)
    env = MultiSegmentWandBVecVideoRecorder(
        env,
        video_folder=f"{log_dir}/videos",
        record_video_trigger=lambda x: x % args.record_freq == 0,
        video_length=args.video_length,
        name_prefix=args.env_name,
    )

    policy = "CnnPolicy" if 'CarRacing' in args.env_name else "MlpPolicy"
    model = create_model(args, env, log_dir, policy)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=args.gradient_save_freq,
        model_save_freq=args.model_save_freq,
        model_save_path=log_dir,
        verbose=2
    )

    model.learn(total_timesteps=args.max_steps, callback=[eval_callback, wandb_cb])

    model.save(f"{log_dir}/final_model")
    wandb.save(f"{log_dir}/final_model.zip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', help='RL algorithm',
                        choices=['DQN', 'A2C', 'PPO', 'SAC', 'TD3'])
    parser.add_argument('--project', type=str, default='ExperimentalSetup', help='Wandb project name')
    parser.add_argument('--env_name', type=str, default='LunarLander-v3', help='Environment name',
                        choices=['FrozenLake-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1',
                                 'Acrobot-v1', 'Pendulum-v1', 'LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--seed', type=int, default=42, help='Seed for the pseudo random generators')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--max_steps', type=int, default=1e6, help='Maximum number of steps')
    parser.add_argument('--eval_freq', type=int, default=5000, help='Frequency of evaluations')
    parser.add_argument('--model_save_freq', type=int, default=5000, help='Frequency of saving the model')
    parser.add_argument('--gradient_save_freq', type=int, default=100, help='Frequency of saving the model')
    parser.add_argument('--record_freq', type=int, default=5000, help='Frequency of recording episodes')
    parser.add_argument('--video_length', type=int, default=1000, help='Length of the recording')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Tags to denote runs')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
