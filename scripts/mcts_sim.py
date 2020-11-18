from minichess.games.abstract.piece import PieceColor
from learning.mcts.mcts import MCTS
from gym_minichess.envs import GardnerMiniChessEnv, RifleMiniChessEnv
import argparse
import os

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')

# ARGS
parser = argparse.ArgumentParser(description='Simulate games with Monte Carlo Tree Search.')

parser.add_argument('--game', dest='game_type', action='store', default='gardner',
                    choices=['gardner', 'dark', 'rifle', 'atomic'],
                    help='minichess rule variant to simulate (default: gardner)')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

# init vars
filepath = 'mcts_data/{}.json'.format(args.game_type)

logging.info('Checking for {} ...'.format(filepath))

if os.path.exists(filepath):
    logging.info('Found.')
else:
    logging.info('Not found. Creating...')

    with open(filepath, 'w+') as f:

        f.write(str(dict()))

env = None

if args.game_type == 'gardner':
    env = GardnerMiniChessEnv
elif args.game_type == 'rifle':
    env = RifleMiniChessEnv
elif args.game_type == 'dark':
    raise NotImplementedError
elif args.game_type == 'atomic':
    raise NotImplementedError

if args.debug: logging.getLogger().setLevel(logging.DEBUG)

m = MCTS.from_json_file(env, filepath)
logging.info('Loaded from {}'.format(filepath))

save_every = 100


try:
    counter = 0
    while True:
        g = env()
        if counter % save_every == 0:
            m.save_to_json(filepath)
            logging.info('Saving at step {}.'.format(counter))

        m.iterate(g, PieceColor.WHITE)
        logging.debug('Completed step {}.'.format(counter))
        counter += 1

except KeyboardInterrupt: # break on keyboard interrupt
    logging.info('Interrupted. Saving...')
    m.save_to_json(filepath)
    logging.info('Saved to {}'.format(filepath))