# Riccati-ZORO
This repository contains an implementation of Riccati-ZORO as described in
> [1]
Riccati-ZORO: An efficient algorithm for heuristic online optimization of internal feedback laws in robust and stochastic model predictive control.\
> Florian Messerer, Yunfan Gao, Jonathan Frey, Moritz Diehl.
> 2025.\
> https://arxiv.org/abs/2511.10473

This implementation uses Python, CasADi and IPOPT.
For the `acados` implementation, we refer to [a section](#acados-implementation) below.


# CasADi implementation

Riccati-ZORO is implemented in the class `SolverRiccatiZoro()` in `src/solver_riccati_zoro.py`.\
As special cases, this class also implements ZORO [2] and SIRO [3].
(ZORO: No Riccati recursion, SIRO: Use the constraint multipliers for the Riccati cost matrices). Each variant can be run with or without gradient correction.

## Requirements

```pip install -r requirements.txt```

The exact package versions used for the paper are documented in `package_versions.txt`.

## Examples
### Towing kite
The example used in the paper.
The towing kite is connected to a ship via a tether of constant length and the objective is to maximize the average thrusting force in order to tow the ship, while respecting a minimum height constraint.
Details are given in [1,2].


### Differential drive robot collision avoidance
A simple nonlinear differential drive robot with state $x=(p_\mathrm{x}, p_\mathrm{y}, \theta, v, \omega)$ consisting of positions $(p_\mathrm{x}, p_\mathrm{y})$, orientation $\theta$, speed $v$, and angular velocity $\omega$.
The controls are the acceleration $a$ and angular acceleration $\alpha$.

### Differential drive robot with uncertain dynamic obstacle 
A differential drive robot needs to avoid a collision with a highly uncertain dynamic obstacle.
By solving tube-OCPs with internal feedback gain optimization, the robot can on purpose correlate its state with the uncertain obstacle prediction in order to improve reference tracking.
This example is described in detail in [5].

# `acados` implementation
A high-performance implementation of Riccati-ZORO is available in `acados`, see
https://github.com/acados/acados.

The exact version used in the paper corresponds to
https://github.com/acados/acados/pull/1676.

This implementation has been mainly developed by [Yunfan Gao](https://github.com/yf-gao/) and [Jonathan Frey](https://github.com/FreyJo/).

## `acados` examples

## Hanging chain benchmark
https://github.com/FreyJo/zoro-NMPC-2021/pull/2

## Differential drive robot collision avoidance
https://github.com/acados/acados/tree/cd290f4/examples/acados_python/zoRO_example/diff_drive

# References

> [2]
An Efficient Algorithm for Tube-based Robust Nonlinear Optimal Control with Optimal Linear Feedback.\
F. Messerer, M. Diehl.
Proceedings of the IEEE Conference on Decision and Control (CDC), 2021.\
https://doi.org/10.1109/CDC45484.2021.9683712;
https://cdn.syscop.de/publications/Messerer2021.pdf

> [3]
Zero-Order Robust Nonlinear Model Predictive Control with Ellipsoidal Uncertainty Sets.\
A. Zanelli, J. Frey, F. Messerer, M. Diehl.
Proceedings of the IFAC Conference on Nonlinear Model Predictive Control (NMPC), 2021.\
https://doi.org/10.1016/j.ifacol.2021.08.523;
https://publications.syscop.de/Zanelli2021.pdf

> [4]
Efficient Zero-Order Robust Optimization for Real-Time Model Predictive Control with acados.\
J. Frey, Y. Gao, F. Messerer, A. Lahr, M. Zeilinger, M. Diehl.
Proceedings of the European Control Conference (ECC), 2024.\
https://doi.org/10.23919/ECC64448.2024.10591148;
https://publications.syscop.de/Frey2024.pdf

> [5]
Stochastic Model Predictive Control with Optimal Linear Feedback for Mobile Robots in Dynamic Environments.\
Y. Gao, F. Messerer, N. van Duijkeren, M. Diehl.
IFAC-PapersOnLine, 2024.\
https://doi.org/10.1016/j.ifacol.2024.09.024;
https://publications.syscop.de/Gao2024.pdf
