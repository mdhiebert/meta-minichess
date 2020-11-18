
# UN-DISTRIBUTED TRAINING

from numpy.lib.twodim_base import mask_indices
from meta_minichess.undist.config import MiniChessConfig

import torch
import torch
import torch.optim as optim
from torch.nn import L1Loss

import numpy as np

class Trainer:
    def __init__(self, config):
        self.config = config

    def adjust_lr(self, optimizer, step_count):
        lr = self.config.lr_init * self.config.lr_decay_rate ** (step_count / self.config.lr_decay_steps)
        lr = max(lr, 0.001)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def sample_batch(self):
        prob_alpha = self.config.priority_prob_alpha
        capacity = self.config.window_size
        batch_size = self.config.batch_size
        beta = self.config.beta

        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        probs = np.array(self.priorities) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.priorities), self.batch_size, p=probs)

        total = len(self.priorities)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        indices = torch.tensor(indices)
        weights = torch.tensor(weights).float()

    def train(self, debug=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = self.config.get_uniform_network().to(device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.config.lr_init, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        target_model = self.config.get_uniform_network().to('cpu')
        target_model.eval()

        for step_num in range(self.config.training_steps):
            lr = self.adjust_lr(optimizer, step_num)


if __name__ == "__main__":
    t = Trainer(MiniChessConfig())
    t.train()