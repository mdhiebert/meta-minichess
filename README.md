# meta-minichess

## Contents

- [Contents](#contents)
- [Quickstart](#quickstart)
- [Setup](#setup)
	- [Automatic](#automatic)
	- [Manual](#manual)
- [To-Do](#to-do)
- [Changelog](#changelog)
- [Objective](#objective)
- [Methodology](#methodology)
	- [Action Space](#action-space)
		- [Legality](#legality)
	- [MCTS](#mcts)
- [Result Log](#result-log)
    - [Naïve Opening](#naïve-opening)
- [References](#references)

## Quickstart

Clone the repository:

```bash
git clone https://github.com/mdhiebert/meta-minichess.git
cd meta-minichess
```

Create conda environment:

```bash
conda env create -f environment.yml
conda activate mmc
```

Launch experiment to train a jack-of-all-trades minichess model:

```bash
python -m learning.alpha_zero.azero
```

See progress in terminal and updated loss plots in `./policy_loss.png` and `./value_loss.png`.

## Setup

Clone the repository:

```bash
git clone https://github.com/mdhiebert/meta-minichess.git
cd meta-minichess
```

### Automatic

Just run the following script to handle setup:

*[TODO]* DOES NOT WORK FOR MACOS

```bash
python scripts/refresh.py --development_dir=<PATH_TO_DEVELOPMENT_DIRECTORY>
```
### Manual

It is recommended to create a conda environment when running meta-minichess games:

```bash
conda create -n mmc python=3.7.4 anaconda
conda activate mmc
```

This repository leverages three separate custom packages: [minichess](https://github.com/mdhiebert/minichess.git), [gym-minichess](https://github.com/mdhiebert/gym-minichess.git), and [muzero-pytorch](https://github.com/mdhiebert/muzero-pytorch.git). To install them for use in `meta-minichess`, execute the following lines in the terminal:


First, we install the underlying minichess libraries: 

```bash
git clone https://github.com/mdhiebert/minichess.git
cd minichess
pip install -e .
cd ..
git clone https://github.com/mdhiebert/gym-minichess
cd gym-minichess
pip install -e .
cd ..
```

Then, we install our MuZero libary:

```bash
git clone https://github.com/mdhiebert/muzero-pytorch
cd muzero-pytorch
pip install -r requirements.txt
pip install torch
pip install tensorboard
pip install -e .
cd ..
```

We can train a model on the standard Gardner ruleset with:
```bash
python main.py --env minichess-gardner-v0 --case minichess --opr train --force
```

## To-Do

_crossed-out = DONE_

- ~~Create a MiniChess library that can handle board state and rules.~~
	- ~~Create from scratch.~~
	- Leverage [existing chess library](https://github.com/niklasf/python-chess).
- ~~Create an Open-AI Gym environment for Gardner MiniChess~~
	- ~~[gym-chess](https://github.com/iamlucaswolf/gym-chess) for reference.~~
	- ~~As well as the [docs](https://github.com/openai/gym/blob/master/docs/creating-environments.md) from [the OpenAI gym repo](https://github.com/openai/gym).~~
- Interface environment with [MuZero](https://github.com/koulanurag/muzero-pytorch) implementation
	- [This](https://github.com/werner-duvaud/muzero-general) is a better documented alternative but is Windows-incompatible.
- Create variant rules as sub-environments in our gym.
	- ~~[Atomic Chess](https://en.wikipedia.org/wiki/Atomic_chess)~~
		- ~~Does not change action space.~~
	- ~~[Dark Chess](https://en.wikipedia.org/wiki/Dark_chess)~~
		- ~~Changes how observations are generated.~~
	- [Extinction Chess](https://en.wikipedia.org/wiki/Extinction_chess)
		- Changes reward function.
	- [Monochromatic Chess](https://en.wikipedia.org/wiki/Monochromatic_chess)
		- Changes legality of moves.
	- [Portal Chess](https://en.wikipedia.org/wiki/Portal_chess)
		- Changes action space.
	- [Progressive Chess](https://en.wikipedia.org/wiki/Progressive_chess)
		- Changes game structure.
	- ~~[Rifle Chess](https://www.chessvariants.com/difftaking.dir/rifle.html)~~
		- ~~Changes action space.~~
- Meta-learn hyperparameters across variable rulesets.
- Update `refresh.py` or other setup scripts to work on other OSs

## Changelog
*[11/18]* Implemented Atomic Chess rule variant.

*[11/15]* Updated `scripts/mcts_sim.py` to support command line arguments to set game.

*[11/14]* Wrote `scripts/mcts_sim.py` to facilitate MCTS simulations.

*[11/12]* Wrote `scripts/refresh.py` to facilitate setup and make development a little easier.

*[11/10]* Implemented Dark Chess rule variant.

*[11/09]* Implemented Rifle Chess rule variant.

*[11/07]* Refactor complete. gym-minichess initial implementation is also complete. Able to run a forked version of [muzero-pytorch](https://github.com/mdhiebert/muzero-pytorch) for out-of-the-box environment. Working on connecting existing environments with MuZero codebase. Seems to have an error running on Windows - will confirm.

Error confirmed for Windows, even with running out-of-box experiments.

*[11/04]* Beginning the refactor. Pushed initital code for [gym-minichess](https://github.com/mdhiebert/gym-minichess). Will do this "bottom-up", starting with MiniChess implementation and build up.

*[11/03]* Decided to create an OpenAI Gym environment to facilitate our RL. Will hopefully be easy to hook it up to a MuZero implementation. We can create several sub-environments within our Gym to handle the variations across rules. Will require a refactor.

*[10/30]* Model able to train, improve, and then achieve ideal end-states (victory vs naive opponents, draw vs. itself) using MCTS and conventional neural network.

## Pseudocode

Michael
1. AlphaZero working
	- Ability to ingest MetaModel or None
2. Implement Extinction
3. Create Skeleton for training

Rishi
1. Look into Cloud Compute & Parallelization
2. Figure out paper off of MetaModel

Tag team skeleton and launch train.py


1. Look into prebuilt AlphaZero (Michael)
2. Look into Cloud Compute and parallelizing selfplay / arena play (Rishi)
3. implement 1 more env (Michael)
4. Meta Model (Rishi)
5. model (Michael)

### Testing

```


## TRAINING
model = Model()
for n in NUM_ITERS: # 1000
	sample g from [atomic, gardner, dark, rifle, extinction]:
		run a full alphazero iteration on g with model

## TESTING
100x:
sample g from [variations]: (same weight)
	JOAT play random
	JOAT play greedy
	plot / log both of the results
we will be satisfied if JOAT wins above some threshold of these games, 60%
JOAT vs random, greedy 20 times given some game state

## META TRAINING
metamodel = MetaModel() # controls dropout

JOAT model (no grad)

Nx:
sample g from [varitions]: (uniform)
	meta = [JOAT + Dropout](https://stackoverflow.com/questions/41583540/custom-dropout-in-tensorflow)
	JOAT x meta: same alphazero iteration. 

	MetaModel:
		outputprobs = [] # differentiable

		r = random.rand()

		binary_str = 0111000, outputprobs

		USE binary_str, calculate backprop with outputprobs
```

## Eval Metrics

Originally:

Have a fixed model, and a model trained on oneshot, pit against each other, show that win rate is >50%

Now:

JOAT Model #1 - do not apply metamodel, no dropout
JOAT Model #2 - we do apply

JOAT1 vs JOAT2, evaluate wins

JOAT1 and JOAT2 vs random, greedy, etc. show superiority

## Objective

TODO

## Methodology

### Action Space

For our RL model, we have chosen to represent our action space as a (1225,) vector. This is because we have:

- 5 x 5 = 25 possible tiles to choose from when selecting a piece
- 8 directions to move it in, maximum magnitude 4 to move from initial position along that direction
- additional 8 possible knight moves
- 3 underpromotions (knight, bishop, rook), with 3 moves to result in an underpromotion (left-diag capture, forward, right-diag capture).

This gives us 5x5x(8x4 + 8 + 3x3) = 1225 possible actions to choose from.

#### Legality

Of course. Not all moves are valid at every step. To account for this, we simply apply a mask over illegal moves to our networks output and re-normalize.

### MCTS

Pseudocode base off of [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py) implementation.
```
game <- game()
net <- net()

Q_sa <- {} # stores Q values for s,a (as defined in the paper)
N_sa <- {} # stores #times edge s,a was visited
N_s  <- {} # stores #times board s was visited
P_s  <- {} # stores initial policy (returned by neural net)
E_s  <- {} # stores game.getGameEnded ended for board s
V_s  <- {} # stores game.getValidMoves for board s

board <- current board

for iteration in NUM_MCTS_ITERATIONS:
	search(board)

counts <- number of times each action was visited from state board

if temp = 0:
	bestAs <- actions with max count
	bestA <- sample(bestAs)

	return onehot of len(ACTION_SPACE) with idx bestA = 1

counts <- [x ** (1. / temp) for x in counts]

return counts / sum(counts) # probabilities


### SEARCH(board)

state <- board

if state not in E_s:
	E_s[state] <- status of game

if state.status != ONGOING:
	return -1 * (value of status)

if s not in P_s:
	P_s[s] <- net(board) # network prediction of board
	valid <- all current legal moves from state
	
	# mask invalid moves
	# renormalize

	if all moves were masked:
		P_s[s] <- (P_s[s] + valids) / P_s[s]

	V_s[s] <- valids

	# pick action with highest upper confidence bound
	for action in actions:
		if action is valid:
			if (s,a) in Q_sa:
				u <- Q_sa[sa] + CPUCT + P_s[s][action] * sqrt(N_s[s]) / (1 + N_sa[sa])
			else:
				u <- CPUCT * P_s[s][action] * sqrt(N_s[s] + eps)

	state <- apply best action to current state

	search(state)

	if (state, action) in Q_sa:
		Q_sa[(s, a)] <- (N_sa[(s, a)] * Q_sa[(s, a)] + v) / (N_sa[(s, a)] + 1)
		N_sa[(s, a)] <- N_sa[(s, a)] + 1

	else:
		Q_sa[(s,a)] <- v
		N_sa[(s,a)] <- 1

	N_s[s] <- N_s + 1

	return -v

```

## Result Log

### Naïve Opening
```
[♜][♞][♝][♛][♚]
[♟][♟][♟][♟][♟]
[ ][ ][ ][ ][ ]
[♙][♙][♙][♙][♙]
[♖][♘][♗][♕][♔]

[♜][♞][♝][♛][♚]
[♟][♟][♟][♟][♟]
[ ][ ][ ][ ][♙]
[♙][♙][♙][♙][ ]
[♖][♘][♗][♕][♔]

[♜][♞][♝][♛][♚]
[♟][♟][♟][ ][♟]
[ ][ ][ ][ ][♟]
[♙][♙][♙][♙][ ]
[♖][♘][♗][♕][♔]

[♜][♞][♝][♛][♚]
[♟][♟][♟][ ][♟]
[ ][ ][♙][ ][♟]
[♙][♙][ ][♙][ ]
[♖][♘][♗][♕][♔]
```

## References

- Learning to Play Minichess Without Human Knowledge - K. Bhuvaneswaran ([paper](https://cs230.stanford.edu/projects_spring_2018/reports/8290438.pdf)) ([code](https://github.com/karthikselva/alpha-zero-general))
    - Very useful reference for training Minichess models, also has some pretrained Keras models.
- Learning to Cope with Adversarial Attacks - X. Lee, A. Heavens et al. ([paper](https://arxiv.org/pdf/1906.12061.pdf))
    - Adversarial RL Grid World algorithm
- Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments - M. Al Shedivat, T. Bansal, et al. ([paper](https://arxiv.org/pdf/1710.03641.pdf)) ([code](https://github.com/openai/robosumo))
	- 'Spider' paper
	- Similar meta-model & outer-loop structure
- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks - C. Finn, P. Abbeel & S. Levine ([paper](https://arxiv.org/pdf/1703.03400.pdf)) ([code](https://github.com/cbfinn/maml))
	- Meta-Learning Architecture
	- Created MAML for Few-Shot Supervised Learning as basline
- Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning - T. Yu, D. Quillen et al. ([paper](https://arxiv.org/pdf/1910.10897v1.pdf))
	- Defined 'Meta-World' as task distribution
	- Utilize as model to define task/rules distribution of different chess piece rule sets