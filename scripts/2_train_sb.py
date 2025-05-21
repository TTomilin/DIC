import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from gym_env.utils import make_env, create_model


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f'{args.algorithm}_{args.env_name}_seed_{args.seed}_{timestamp}'
    base_path = Path(__file__).parent.parent.resolve()
    log_dir = f"{base_path}/logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(lambda: make_env(args.env_name, seed=args.seed), n_envs=args.n_envs)

    policy = "CnnPolicy" if 'CarRacing' in args.env_name else "MlpPolicy"
    model = create_model(args, env, log_dir, policy)

    model.learn(total_timesteps=args.max_steps)

    model.save(f"{log_dir}/final_model")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', help='RL algorithm',
                        choices=['DQN', 'A2C', 'PPO', 'SAC', 'TD3'])
    parser.add_argument('--project', type=str, default='DQN Demo', help='Wandb project name')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name',
                        choices=['FrozenLake-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1',
                                 'Acrobot-v1', 'Pendulum-v1', 'LunarLander-v3', 'CarRacing-v3'])
    parser.add_argument('--seed', type=int, default=42, help='Seed for the pseudo random generators')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--max_steps', type=int, default=1e5, help='Maximum number of steps')
    parser.add_argument('--model_save_freq', type=int, default=5000, help='Frequency of saving the model')
    parser.add_argument('--gradient_save_freq', type=int, default=100, help='Frequency of saving the model')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
