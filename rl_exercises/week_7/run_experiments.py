import os

import gymnasium as gym
import hydra
import pandas as pd
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rnd_dqn import RNDDQNAgent


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # set up results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # initiate DQN agent with only epsilon-greedy exploration
    agentEpsilon = DQNAgent(
        env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
    )

    # train DQN agent
    frames_Epsilon, average_returns_Epsilon, std_returns_Epsilon = agentEpsilon.train(
        num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval
    )

    # save results
    dqn_data = pd.DataFrame(
        {
            "frames_Epsilon": frames_Epsilon,
            "AvgRewardEpsilon(10)": average_returns_Epsilon,
            "StdRewardEpsilon(10)": std_returns_Epsilon,
        }
    )

    dqn_data.to_csv(
        os.path.join(results_dir, f"dqn_training_data_seed_{cfg.seed}.csv"), index=False
    )

    # initiate RND_DQN agent
    agentRND = RNDDQNAgent(
        env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        rnd_hidden_size=cfg.agent.rnd_hidden_size,
        rnd_lr=cfg.agent.rnd_lr,
        rnd_update_freq=cfg.agent.rnd_update_freq,
        rnd_n_layers=cfg.agent.rnd_n_layers,
        rnd_reward_weight=cfg.agent.rnd_reward_weight,
    )

    # train RND_DQN agent
    frames_RND, average_returns_RND, std_returns_RND = agentRND.train(
        num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval
    )

    # save results
    rnd_data = pd.DataFrame(
        {
            "frames_RND": frames_RND,
            "AvgRewardRND(10)": average_returns_RND,
            "StdRewardRND(10)": std_returns_RND,
        }
    )

    rnd_data.to_csv(
        os.path.join(results_dir, f"rnd_dqn_training_data_seed_{cfg.seed}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
