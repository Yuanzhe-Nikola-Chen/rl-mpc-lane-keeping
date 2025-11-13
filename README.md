# 🏭 Industry Context

This project is not affiliated with Mobileye, but it explores planning and control principles commonly evaluated in modern camera-centric ADAS/AV systems, including approaches used by leading companies such as Mobileye, Waymo, Tesla, and NVIDIA. The comparison of MPC and RL under a unified vehicle model mirrors the type of internal benchmarking often performed in real-world autonomous driving research teams.

# RL vs MPC Lane-Keeping Planner

This repository implements a simple autonomous driving lane-keeping scenario
using a kinematic bicycle model. We compare:

- **Model Predictive Control (MPC)**: optimization-based controller
- **Proximal Policy Optimization (PPO)**: reinforcement learning-based policy

The goal is to highlight trade-offs between classical control and deep RL
for autonomous driving planning and control.

## Features

- Kinematic bicycle vehicle model
- Lane-keeping environment (Gym-style API)
- Random-shooting MPC controller
- PPO agent implemented in PyTorch
- Scripts to train RL, run MPC, and compare both

## Install

```bash
pip install -r requirements.txt
```

## Run MPC demo

```bash
python scripts/run_mpc.py
```

## Train PPO

```bash
python scripts/train_ppo.py
```

## Compare RL vs MPC

```bash
python scripts/compare_mpc_vs_ppo.py
```

## Roadmap

- Add obstacles and collision penalties
- Add risk/safety metrics (RSS-inspired)
- Extend to multi-lane / lane change scenarios
