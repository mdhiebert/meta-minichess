import argparse
import subprocess
import os
from os import path
import logging

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Install/Update dependencies for meta-minichess experiments.')

parser.add_argument('--development_dir', dest='dev_dir', action='store', default='.',
                    help='top-level directory where development will occur (default: current directory)')

# parse out development directory (if provided)
args = parser.parse_args()
dev_dir = path.expanduser(args.dev_dir)

logging.info('Using Development Directory {}'.format(dev_dir))
logging.info('Absolute path: {}'.format(path.abspath(dev_dir)))

def refresh_dir(d):
    logging.info('Checking for minichess...')
    # check for d:
    if path.isdir(path.join(dev_dir, d)):
        logging.info('Found {}. Pulling...'.format(d))
        subprocess.run(['git', 'pull'], shell=True, cwd=path.join(dev_dir, d))
    else:
        logging.info('Cloning {}...'.format(d))
        subprocess.run(['git', 'clone', 'https://github.com/mdhiebert/{}.git'.format(d)], shell=True, cwd=path.abspath(dev_dir))

    logging.info('Success.')
    logging.info('Installing {} library...'.format(d))

    subprocess.run(['pip', 'install', '-e', '.'], shell=True, cwd=path.join(dev_dir, d))    
    logging.info('Success.')


refresh_dir('minichess')
refresh_dir('gym-minichess')


