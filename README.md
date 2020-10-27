# meta-minichess

## Contents

- [Contents](#Contents)
- [Changelog](#changelog)
- [Objective](#objective)
- [Methodology](#methodology)
- [Result Log](#result-log)
    - [Naïve Opening](#naïve-opening)
- [References](#references)

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