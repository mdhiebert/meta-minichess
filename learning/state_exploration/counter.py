import json
from learning.state_exploration.explore_arena import ExploreArena

from games.gardner import GardnerMiniChessGame
from games.baby.BabyChessGame import BabyChessGame
from games.mallet import MalletChessGame
from games.rifle import RifleChessGame
from games.atomic import AtomicChessGame

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

CLASS_TO_IDX = {
    'GardnerMiniChessGame': 0,
    'BabyChessGame': 1,
    'MalletChessGame': 2,
    'RifleChessGame': 3,
    'AtomicChessGame': 4
}

EMPTY = [False] * 5

class Counter:
    def __init__(self, json_path):
        self.json_path = json_path
        #print('Loading results...')
        self.state_dict = self._load_json(self.json_path)

        self.arena = ExploreArena()

        self.games = [
            GardnerMiniChessGame(),
            BabyChessGame(),
            MalletChessGame(),
            RifleChessGame(),
            AtomicChessGame()
        ]

        self.game_counter = []

        self.gardnerlens = []
        self.babylens = []
        self.malletlens = []
        self.riflelens = []
        self.atomiclens = []

        # PLOT
        self.fig = plt.figure(figsize=(17, 7))


    def _init_plots(self):
        self.spec = gridspec.GridSpec(ncols=4, nrows=2)

        self.total = self.fig.add_subplot(self.spec[0, :2])
        self.total.set_title('Total Games Played')
        self.gamesplt = self.fig.add_subplot(self.spec[1, :2])
        self.gamesplt.set_title('Unique States Seen by Variant')
        self.gamesplt.set_xlabel('Iterations')
        self.gamesplt.set_ylabel('States')
        self.intersection = self.fig.add_subplot(self.spec[:, 2:])

        self.total.plot(list(range(len(self.game_counter))), self.game_counter, c='k')
        self.gamesplt.plot(list(range(len(self.gardnerlens))), self.gardnerlens, c='b', label='Gardner')
        self.gamesplt.plot(list(range(len(self.babylens))), self.babylens, c='g', label='Baby')
        self.gamesplt.plot(list(range(len(self.malletlens))), self.malletlens, c='m', label='Mallet')
        self.gamesplt.plot(list(range(len(self.riflelens))), self.riflelens, c='r', label='Rifle')
        self.gamesplt.plot(list(range(len(self.atomiclens))), self.atomiclens, c='c', label='Atomic')
        self.gamesplt.legend(loc='upper left')


        return self.total, self.gamesplt, self.intersection
    
    def _make_heatmap(self, ax, mat):
        ax.clear()
        games = ['gardner', 'baby', 'mallet', 'rifle', 'atomic']

        inter = mat

        im = ax.imshow(inter, cmap='cividis')

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(games)))
        ax.set_yticks(np.arange(len(games)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(games)
        ax.set_yticklabels(games)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(games)):
            for j in range(len(games)):
                if i == j:
                    text = ax.text(j, i, '---', ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, f'{inter[i, j]:.1e}',
                            ha="center", va="center", color="w")

        ax.set_title("IOU of State Space")
        # fig.tight_layout()

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def _save_json(self):
        with open(self.json_path, 'w+') as f:
            json.dump(self.state_dict, f)

    def start(self, i, quit_after=None, save_every=-1):

        QUIT_AFTER = -1 if quit_after is None else quit_after

        c = 0
        while True:
            c += 1

            if QUIT_AFTER != -1 and c > QUIT_AFTER: break

            if save_every != -1 and c % save_every == 0:
                self.execute(True)
            else:
                self.execute()

            break

        int_mat = self.metrics()

        self._make_heatmap(self.intersection, int_mat)

        self.total.plot(list(range(len(self.game_counter))), self.game_counter, c='k')
        self.gamesplt.plot(list(range(len(self.gardnerlens))), self.gardnerlens, c='b', label='Gardner')
        self.gamesplt.plot(list(range(len(self.babylens))), self.babylens, c='g', label='Baby')
        self.gamesplt.plot(list(range(len(self.malletlens))), self.malletlens, c='m', label='Mallet')
        self.gamesplt.plot(list(range(len(self.riflelens))), self.riflelens, c='r', label='Rifle')
        self.gamesplt.plot(list(range(len(self.atomiclens))), self.atomiclens, c='c', label='Atomic')

        return self.total, self.gamesplt, self.intersection
        # try:

        # except KeyboardInterrupt:
        #     self.close()
        # finally:
        #     self.close()

    def execute(self, save=False):

        notional_args = {
            'numWorkers': 6,
            'arenaComparePerGame': 40,
            'maxMoves': 200
        }

        results = self.arena.playGames('random', 'random', notional_args, self.games)

        #print('Compiling results')

        for result in results:
            for s,c in result:
                c_idx = CLASS_TO_IDX[c]

                if s in self.state_dict:
                    self.state_dict[s][c_idx] = True
                else:
                    l = EMPTY.copy()
                    l[c_idx] = True

                    self.state_dict[s] = l
        
        if save: self._save_json()

        if len(self.game_counter) == 0:
            self.game_counter = [notional_args['arenaComparePerGame'] * 5]
        else:
            self.game_counter.append(self.game_counter[-1] + notional_args['arenaComparePerGame'] * 5)


    def close(self):
        #print('Saving results...')
        self._save_json()
        #print('Saved.')

        self.metrics()

    def metrics(self):
        #print('Collecting sets.')
        gardner = set()
        baby = set()
        mallet = set()
        rifle = set()
        atomic = set()

        sets = [gardner, baby, mallet, rifle, atomic]
        snames = ['gardner', 'baby', 'mallet', 'rifle', 'atomic']

        for state in self.state_dict:
            for s,b in zip(sets, self.state_dict[state]):
                if b: s.add(state)

        self.gardnerlens.append(len(gardner))
        self.babylens.append(len(baby))
        self.malletlens.append(len(mallet))
        self.riflelens.append(len(rifle))
        self.atomiclens.append(len(atomic))

        #print('Computing metrics...')

        int_mat = np.zeros((5, 5), np.float)

        row = 0
        for s1,n1 in zip(sets, snames):
            col = 0
            for s2,n2 in zip(sets, snames):
                lint = len(s1.intersection(s2))
                luni = len(s1.union(s2))

                iou = float(lint) / float(luni)

                int_mat[row,col] = iou

                if row==col: int_mat[row,col] = 0

                # #print(f'{n1} <-> {n2} :: S1: {len(s1)} / S2: {len(s2)} / I: {lint} / U: {luni} / IOU: {iou}')
                col += 1
            row += 1

        return int_mat



        