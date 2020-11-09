# meta-minichess

## Contents

- [Contents](#contents)
- [Setup](#setup)
- [To-Do](#to-do)
- [Changelog](#changelog)
- [Objective](#objective)
- [Methodology](#methodology)
	- [Action Space](#action-space)
		- [Legality](#legality)
- [Result Log](#result-log)
    - [Naïve Opening](#naïve-opening)
- [References](#references)

## Setup

Clone the repository:

```
git clone https://github.com/mdhiebert/meta-minichess.git
```

This repository leverages three separate custom packages: [minichess](https://github.com/mdhiebert/minichess.git), [gym-minichess](https://github.com/mdhiebert/gym-minichess), and [muzero-general]() . To install them for use in `meta-minichess`, execute the following lines in the terminal:

```
git clone https://github.com/mdhiebert/minichess.git
cd minichess
pip install -e .
cd ..
git clone https://github.com/mdhiebert/gym-minichess
cd gym-minichess
pip install -e .
cd ..
git clone https://github.com/mdhiebert/muzero-pytorch
cd muzero-pytorch
pip install -r requirements.txt
pip install ray
pip install -e .
cd ..
```

We can train a model on the standard Gardner ruleset with:
```
python main.py --env gym-minichess:minichess-gardner-v0 --case minichess --opr train --force
```

## To-Do

_crossed-out = DONE_

- Create a MiniChess library that can handle board state and rules.
	- ~~Create from scratch.~~
	- Leverage [existing chess library](https://github.com/niklasf/python-chess).
- ~~Create an Open-AI Gym environment for Gardner MiniChess~~
	- ~~[gym-chess](https://github.com/iamlucaswolf/gym-chess) for reference.~~
	- ~~As well as the [docs](https://github.com/openai/gym/blob/master/docs/creating-environments.md) from [the OpenAI gym repo](https://github.com/openai/gym).~~
- Interface environment with [MuZero](https://github.com/koulanurag/muzero-pytorch) implementation
	- [This](https://github.com/werner-duvaud/muzero-general) is a better documented alternative but is Windows-incompatible.
- Create variant rules as sub-environments in our gym.
	- [Atomic Chess](https://en.wikipedia.org/wiki/Atomic_chess)
		- Does not change action space.
	- [Dark Chess](https://en.wikipedia.org/wiki/Dark_chess)
		- Changes how observations are generated.
	- [Extinction Chess](https://en.wikipedia.org/wiki/Extinction_chess)
		- Changes reward function.
	- [Monochromatic Chess](https://en.wikipedia.org/wiki/Monochromatic_chess)
		- Changes legality of moves.
	- [Portal Chess](https://en.wikipedia.org/wiki/Portal_chess)
		- Changes action space.
	- [Progressive Chess](https://en.wikipedia.org/wiki/Progressive_chess)
		- Changes game structure.
	- [Rifle Chess](https://www.chessvariants.com/difftaking.dir/rifle.html)
		- Changes action space.
- Meta-learn hyperparameters across variable rulesets.

## Changelog

*[11/07]* Refactor complete. gym-minichess initial implementation is also complete. Able to run a forked version of [muzero-pytorch](https://github.com/mdhiebert/muzero-pytorch) for out-of-the-box environment. Working on connecting existing environments with MuZero codebase. Seems to have an error running on Windows - will confirm.

*[11/04]* Beginning the refactor. Pushed initital code for [gym-minichess](https://github.com/mdhiebert/gym-minichess). Will do this "bottom-up", starting with MiniChess implementation and build up.

*[11/03]* Decided to create an OpenAI Gym environment to facilitate our RL. Will hopefully be easy to hook it up to a MuZero implementation. We can create several sub-environments within our Gym to handle the variations across rules. Will require a refactor.

*[10/30]* Model able to train, improve, and then achieve ideal end-states (victory vs naive opponents, draw vs. itself) using MCTS and conventional neural network.

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

TODO

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