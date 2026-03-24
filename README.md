# RL vs MPC for Lane-Keeping: A Control-Oriented Study

This project investigates the trade-offs between model-based control and learning-based approaches for autonomous driving, using a lane-keeping task under a unified vehicle model.

Specifically, it compares:

- **Model Predictive Control (MPC)**: a constraint-aware optimization-based controller  
- **Proximal Policy Optimization (PPO)**: a learning-based policy trained from interaction  

The goal is not only performance comparison, but to understand:

- How control constraints influence safety and stability  
- When learning-based policies can complement model-based control  
- Trade-offs between interpretability, robustness, and adaptability  

---

## Method

The system is built on a kinematic bicycle model and a simplified lane-keeping environment.

Key components include:

- **Vehicle dynamics**: kinematic bicycle model  
- **MPC controller**: random-shooting optimization over a finite horizon  
- **RL agent**: PPO implemented in PyTorch  
- **Evaluation**: trajectory tracking performance and qualitative behavior comparison  

Both approaches operate under the same vehicle model and environment to enable a fair comparison.

---

## Features

- Unified vehicle model for both MPC and RL  
- Lane-keeping environment with Gym-style API  
- Optimization-based MPC controller  
- PPO agent for policy learning  
- Comparative evaluation scripts  

---

## Research Context

This project is an academic exploration of planning and control methods in autonomous driving.

It reflects common research questions in modern autonomous systems:

- How to combine model-based control with learning-based approaches  
- How to ensure safety under constraints  
- How to balance optimality and real-time performance  

The structure of this comparison is inspired by system-level evaluations in real-world autonomous driving research.

---

## Installation

pip install -r requirements.txt

---

## Usage

Run MPC:

python scripts/run_mpc.py

Train PPO:

python scripts/train_ppo.py

Compare both approaches:

python scripts/compare_mpc_vs_ppo.py

---

## Future Work

- Incorporate obstacle avoidance and collision constraints  
- Introduce safety-aware metrics (e.g., RSS-inspired constraints)  
- Extend to multi-lane and lane-change scenarios  
- Explore hybrid MPC + learning architectures  

---

## Notes

This project is intended as a control-oriented study of autonomous driving systems.  
The focus is on understanding system-level trade-offs between optimization-based and learning-based approaches, rather than achieving state-of-the-art performance.

---

## License

MIT License
