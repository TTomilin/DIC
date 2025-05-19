import argparse
import torch
from gymnasium import make
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic


def make_env(task, seed=0):
    def _init():
        env = make(task)
        env.reset(seed=seed)
        return env
    return _init


def get_envs(env_name, num_envs=8, seed=0):
    train_envs = SubprocVectorEnv([make_env(env_name, i + seed) for i in range(num_envs)])
    test_envs = SubprocVectorEnv([make_env(env_name, i + num_envs + seed) for i in range(num_envs)])
    return train_envs, test_envs


def setup_network_and_policy(env, lr=1e-4, hidden_sizes=[128, 128]):
    # Discrete MountainCar
    state_shape = env.observation_space[0].shape or env.observation_space[0].n
    action_shape = env.action_space[0].n
    net = Net(state_shape, hidden_sizes=hidden_sizes, device='cuda')
    actor = Actor(net, action_shape, device='cuda').to('cuda')
    critic = Critic(net, device='cuda').to('cuda')
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    # PPO for discrete actions uses Categorical
    policy = PPOPolicy(
        actor, critic, actor_optim, critic_optim,
        torch.distributions.Categorical,
        discount_factor=0.99, max_grad_norm=0.5,
        eps_clip=0.2, vf_coef=0.5, ent_coef=0.01,
        reward_normalization=False, dual_clip=None,
        gae_lambda=0.95, value_clip=True
    )
    return policy


def main(cfg):
    train_envs, test_envs = get_envs(cfg.env_name, cfg.num_envs)

    policy = setup_network_and_policy(train_envs, cfg.learning_rate, cfg.hidden_sizes)

    # No replay buffer for PPO
    train_collector = Collector(policy, train_envs)
    test_collector = Collector(policy, test_envs)

    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=cfg.max_epoch,  # total training epochs
        step_per_epoch=cfg.step_per_epoch,
        repeat_per_collect=10,  # how many times we update per data collection
        episode_per_test=cfg.episode_per_test,
        batch_size=cfg.batch_size
    )

    print(f'Finished PPO training! Results: {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--env_name', type=str, default='MountainCar-v0')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--episode_per_test', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
