from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.policy_params import DQNParams
from tianshou.highlevel.trainer import (
    EpochStopCallbackRewardThreshold,
    EpochTestCallbackDQNSetEps,
    EpochTrainCallbackDQNSetEps,
)
from tianshou.utils.logging import run_main


def main() -> None:
    experiment = (
        DQNExperimentBuilder(
            EnvFactoryRegistered(task="CartPole-v1", seed=0, venv_type=VectorEnvType.DUMMY),
            ExperimentConfig(
                persistence_enabled=True,
                watch=False,
                watch_render=1 / 35,
                watch_num_episodes=1,
            ),
            SamplingConfig(
                num_epochs=10,
                step_per_epoch=10000,
                batch_size=64,
                num_train_envs=5,
                num_test_envs=3,
                buffer_size=20000,
                step_per_collect=10,
                update_per_step=1 / 10,
            ),
        )
        .with_dqn_params(
            DQNParams(
                lr=1e-3,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            ),
        )
        .with_model_factory_default(hidden_sizes=(64, 64))
        .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
        .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(500))
        .with_logger_factory(LoggerFactoryDefault(logger_type="wandb", wandb_project="FreshProject"))
        .build()
    )
    experiment.run()


if __name__ == "__main__":
    run_main(main)
