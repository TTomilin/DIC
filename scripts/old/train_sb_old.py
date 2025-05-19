import argparse

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

algorithms = {
    'PPO': PPO,
    'A2C': A2C,
    'DQN': DQN,
}


def main(args: argparse.Namespace) -> None:
    # Create and wrap the environment
    env = make_vec_env(args.env_name, n_envs=args.num_envs)

    # Define the model
    algorithm = algorithms[args.algorithm]
    model = algorithm("MlpPolicy", env, learning_rate=args.learning_rate, gamma=args.gamma, verbose=1, device="cuda")

    # Callbacks
    eval_env = gym.make(args.env_name)
    eval_callback = EvalCallback(eval_env, best_model_save_path='../../logs/',
                                 log_path='../../logs/', eval_freq=args.eval_freq,
                                 deterministic=True, render=False,
                                 n_eval_episodes=args.n_eval_episodes)

    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='../../logs/', name_prefix='model')

    # Start the training
    model.learn(total_timesteps=args.max_steps, log_interval=args.log_interval,
                callback=[eval_callback, checkpoint_callback])

    # Close the environment
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['DQN', 'A2C', 'PPO'], help='RL algorithm')
    parser.add_argument('--project', type=str, default='ExperimentalSetup', help='Wandb project name')
    parser.add_argument('--num_envs', type=int, default=10, help='Number of parallel environments')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy_kwargs', type=str, default='dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])',
                        help='Policy kwargs for DQN')
    parser.add_argument('--max_steps', type=int, default=4e5, help='Maximum number of steps')
    parser.add_argument('--eval_freq', type=int, default=250, help='Frequency of evaluations')
    parser.add_argument('--save_freq', type=int, default=5e4, help='Frequency of saving the model')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of episodes per evaluation')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--reward_threshold', type=float, default=400, help='Reward threshold to stop training')
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
