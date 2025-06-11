"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

from collections import OrderedDict

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks

        # Size of input and output layers
        obs_dim = env.observation_space.shape[0]

        # RND hyperparameters
        self.rnd_hidden_size = rnd_hidden_size
        self.rnd_n_layers = rnd_n_layers
        self.rnd_lr = rnd_lr
        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        # Define the fixed network
        self.fixed_network = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(obs_dim, self.rnd_hidden_size)),
                    ("relu1", nn.ReLU()),
                    *[
                        (
                            f"fc{i + 2}",
                            nn.Linear(self.rnd_hidden_size, self.rnd_hidden_size),
                        )
                        for i in range(self.rnd_n_layers - 1)
                    ],
                    ("out", nn.Linear(self.rnd_hidden_size, self.rnd_hidden_size)),
                ]
            )
        )

        # Freeze weights of the fixed network
        for param in self.fixed_network.parameters():
            param.requires_grad = False

        # Define the predictor network
        self.predictor_network = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(obs_dim, self.rnd_hidden_size)),
                    ("relu1", nn.ReLU()),
                    *[
                        (
                            f"fc{i + 2}",
                            nn.Linear(self.rnd_hidden_size, self.rnd_hidden_size),
                        )
                        for i in range(self.rnd_n_layers - 1)
                    ],
                    ("out", nn.Linear(self.rnd_hidden_size, self.rnd_hidden_size)),
                ]
            )
        )

        # Optimizer for predictor only
        self.rnd_optimizer = optim.Adam(
            self.predictor_network.parameters(), lr=self.rnd_lr
        )

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        # TODO: compute the MSE
        # TODO: update the RND network
        states, _, _, next_states, _, _ = zip(*training_batch)

        s = torch.tensor(np.array(states), dtype=torch.float32)
        # s_next = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Learned embeddings from the predictor network
        pred_output = self.predictor_network(s)

        # Random but fixed embeddings from the fixed random network
        with torch.no_grad():
            fixed_random_output = self.fixed_network(s)

        # Compute the MSE
        loss = nn.MSELoss()(pred_output, fixed_random_output)

        # Update the predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        # TODO: get error

        # Learned embeddings from the predictor network
        s = torch.tensor(np.array([state]), dtype=torch.float32)
        pred_output = self.predictor_network(s)

        # Random but fixed embeddings from the fixed random network
        with torch.no_grad():
            fixed_random_output = self.fixed_network(s)

        # Compute the bonus
        rnd_bonus = nn.MSELoss()(pred_output, fixed_random_output)

        # Scale the bonus
        scaled_bonus = rnd_bonus.item() * self.rnd_reward_weight

        return scaled_bonus

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        # episode_rewards = []
        # steps = []

        # create lists of average rewards and frames for plotting
        frames: List[int] = []
        average_returns: List[float] = []
        std_returns: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            reward += self.get_rnd_bonus(next_state)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update agent if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                # update RND network
                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                # episode_rewards.append(ep_reward)
                # steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    std = np.std(recent_rewards[-10:])
                    frames.append(frame)
                    average_returns.append(avg)
                    std_returns.append(std)
                    print(
                        f"Frame {frame}, AvgRewardRND(10): {avg:6.1f} ± {std:4.1f} ε={self.epsilon():.3f}"
                    )

        return frames, average_returns, std_returns


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent

    # initiate agent
    agent = RNDDQNAgent(
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

    frames, average_returns_RND = agent.train(
        num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval
    )

    """
    frames, average_returns_RND = agentRND.train(
        num_frames=cfg.train.num_frames,
        eval_interval=cfg.train.eval_interval
    )

    frames, average_returns_Epsilon = agentepsilon.train(
        num_frames=cfg.train.num_frames,
        eval_interval=cfg.train.eval_interval
    )

    # Plot average return vs. steps using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(
        frames,
        average_returns,
        label=f"DQN with RND (Seed: {cfg.seed})",
    )
    plt.xlabel("Frames")
    plt.ylabel("Average Return")
    plt.title(
        f"Average Return vs. Frames: DQN with RND (Seed: {cfg.seed})"
    )
    plt.grid(True)
    plt.legend()
    # Save the plot as PNG
    plot_path = os.path.join(
        os.path.dirname(__file__), "plots", "average_return_vs_frames.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")


    # Saving to .csv for simplicity
    # Could also be e.g. npz
    training_data = pd.DataFrame({"frames": frames, "AvgReward(10)": average_returns})
    training_data.to_csv(f"training_data_seed_{cfg.seed}.csv", index=False)
    """


if __name__ == "__main__":
    main()
