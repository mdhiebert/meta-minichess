from collections import deque

import numpy as np

from learning.muzero.game import Game, Action


class MiniChessWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount)
        self.k = k
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [Action(int(idx)) for idx in self.env.legal_actions()]

    def step(self, action):
        # action is an int representing an action ID
        obs, reward, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.obs(len(self.rewards)), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):
            self.obs_history.append(obs)

        return self.obs(0)

    def obs(self, i):
        frames = self.obs_history[i:i + self.k]
        return np.array(frames).flatten()

    def close(self):
        self.env.close()
