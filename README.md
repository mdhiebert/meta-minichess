# meta-minichess

## Contents

- [Contents](#Contents)
- [To-Do](#to-do)
- [Changelog](#changelog)
- [Objective](#objective)
- [Methodology](#methodology)
- [Result Log](#result-log)
    - [Naïve Opening](#naïve-opening)
- [References](#references)

## To-Do

_crossed-out = DONE_

- Create a MiniChess library that can handle board state and rules.
	- ~~Create from scratch.~~
	- Leverage [existing chess library](https://github.com/niklasf/python-chess).
- Create an Open-AI Gym environment for Gardner MiniChess
	- [gym-chess](https://github.com/iamlucaswolf/gym-chess) for reference.
	- As well as the [docs](https://github.com/openai/gym/blob/master/docs/creating-environments.md) from [the OpenAI gym repo](https://github.com/openai/gym).
- Interface environment with [MuZero](https://github.com/koulanurag/muzero-pytorch) implementation
	- [This](https://github.com/werner-duvaud/muzero-general) is a better documented alternative but is Windows-incompatible.
- Create variant rules as sub-environments in our gym.
	- [Atomic Chess](https://en.wikipedia.org/wiki/Atomic_chess)
	- [Dark Chess](https://en.wikipedia.org/wiki/Dark_chess)
	- [Extinction Chess](https://en.wikipedia.org/wiki/Extinction_chess)
	- [Monochromatic Chess](https://en.wikipedia.org/wiki/Monochromatic_chess)
	- [Portal Chess](https://en.wikipedia.org/wiki/Portal_chess)
	- [Progressive Chess](https://en.wikipedia.org/wiki/Progressive_chess)
	- [Rifle Chess](https://www.chessvariants.com/difftaking.dir/rifle.html)
- Meta-learn hyperparameters across variable rulesets.

## Changelog

TODO

## Objective

TODO

## Methodology

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