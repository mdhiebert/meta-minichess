import os
import sys
import time

import numpy as np
from tqdm import tqdm

from learning.alpha_zero.distributed.utils import *
from learning.alpha_zero.distributed.pytorch.NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

args = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

LEN_ACTION_SPACE = 1225

class MetaMCNet(NeuralNet):
    '''
        A Meta Net which takes in an MCGardnerNet and modifies hyperparameters and applies
        controlled dropout to 
    '''
    def __init__(self, game, mcnet):
        self.nnet = mcnet
        self.game = game
        self.board_x, self.board_y = (5, 5)
        self.action_size = self.game.getActionSize()

        # process board
        self.conv1 = nn.Conv2d(1, args['num_channels'], 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1)
        self.conv4 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args['num_channels'])
        self.bn2 = nn.BatchNorm2d(args['num_channels'])
        self.bn3 = nn.BatchNorm2d(args['num_channels'])
        self.bn4 = nn.BatchNorm2d(args['num_channels'])

        self.to_hidden = nn.Linear(args['num_channels']*(self.board_x-4)*(self.board_y-4), 1024)
        self.th_bn1 = nn.BatchNorm1d(1024)

        self.fc1dropout = nn.Linear(1024, self.nnet.nnet.fc1.in_features)
        self.fc2dropout = nn.Linear(1024, self.nnet.nnet.fc2.in_features)
        self.fc3dropout = nn.Linear(1024, self.nnet.nnet.fc3.in_features)
        self.fc4dropout = nn.Linear(1024, self.nnet.nnet.fc4.in_features)

    def steep_sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-20 * x))

    def compute_meta_hidden(self, x):
        # x: batch_size x board_x x board_y
        x = x.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        x = F.relu(self.bn1(self.conv1(x)))                          # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn2(self.conv2(x)))                          # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn3(self.conv3(x)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        x = F.relu(self.bn4(self.conv4(x)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        x = x.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        return self.th_bn1(self.to_hidden(x))

    def get_dropouts(self, x):
        hidden = self.compute_meta_hidden(x)

        x = self.compute_meta_hidden(x)

        dropout1 = self.steep_sigmoid(self.fc1dropout(x))
        dropout2 = self.steep_sigmoid(self.fc2dropout(x))
        dropout3 = self.steep_sigmoid(self.fc3dropout(x))
        dropout4 = self.steep_sigmoid(self.fc4dropout(x))

        return dropout1, dropout2, dropout3, dropout4

    def compute_net_hidden(self, x):
        with torch.no_grad():
            # x: batch_size x board_x x board_y
            x = x.view(-1, 1, self.board_x, self.board_y) # batch_size x 1 x board_x x board_y
            x = F.relu(self.bn1(self.conv1(x))) # batch_size x num_channels x board_x x board_y
            x = F.relu(self.bn2(self.conv2(x))) # batch_size x num_channels x board_x x board_y
            x = F.relu(self.bn3(self.conv3(x))) # batch_size x num_channels x (board_x-2) x (board_y-2)
            x = F.relu(self.bn4(self.conv4(x))) # batch_size x num_channels x (board_x-4) x (board_y-4)
            x = x.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

            return x

    def forward_with_dropout(self, x):
        self.nnet.nnet.requires_grad = False
        net_x = self.compute_net_hidden(x)
        d1,d2,d3,d4 = self.get_dropouts(x)

        # dropout1
        x = F.relu(self.bn1(self.nnet.nnet.fc1(d1 * net_x)))
        x = F.relu(self.bn2(self.nnet.nnet.fc2(d2 * x)))
        pi = F.relu(self.bn3(self.nnet.nnet.fc3(d3 * x)))
        v = F.relu(self.bn4(self.nnet.nnet.fc4(d4 * x)))

        return pi, v

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.nnet.parameters())

        losses = []

        for epoch in range(args['epochs']):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args['batch_size'])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args['cuda']:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.forward_with_dropout(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            losses.append((pi_losses.avg,v_losses.avg))
        return losses

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        board = np.array(board)
        board = board[np.newaxis, :, :]
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args['cuda']: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.nnet.eval()
        with torch.no_grad():
            pi, v = self.forward_with_dropout(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='metacheckpoint', filename='metacheckpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='metacheckpoint', filename='metacheckpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.nnet.load_state_dict(checkpoint['state_dict'])

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict: del self_dict['pool']
        return self_dict
