
# Multi-Agent Reinforcement Learning for Cooperative CPU and Memory Resource Optimization

This repository provides a simulation and learning environment for **multi-agent reinforcement learning (MARL)** that optimizes CPU and memory resources cooperatively.  
Agents represent CPU cores and memory controllers that share global and local observations to achieve energy-efficient, low-latency task execution.

## Features
- Multi-agent environment: each agent controls one CPU core or a memory unit.
- Decentralized partially observable MDP (Dec-POMDP) formulation.
- Training with MADDPG and QMIX algorithms.
- Dynamic workloads and contention modeling for shared memory buses.
- Metrics for energy efficiency, latency, throughput, and fairness.

---
## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train using MADDPG
python -m experiments.train_maddpg --config configs/maddpg.yaml

# Evaluate trained agents
python -m experiments.evaluate --checkpoint artifacts/maddpg/best.pt --episodes 30
```

---
## Directory Structure
```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── maddpg.yaml
│   └── qmix.yaml
├── src/
│   ├── env.py
│   ├── agents/
│   │   ├── maddpg.py
│   │   ├── qmix.py
│   │   └── replay_buffer.py
│   ├── utils.py
│   └── evaluate.py
├── experiments/
│   ├── train_maddpg.py
│   └── evaluate.py
├── scripts/
│   └── plot_metrics.py
├── docs/
│   ├── methodology.md
│   └── results_template.md
└── tests/
    └── test_env.py
```
---
## Core Idea
Each CPU agent learns a policy \(\pi_i(a_i|o_i)\) to allocate CPU cycles, and each memory agent learns a policy for bandwidth or caching decisions. Cooperation is enforced via a global reward combining latency, energy, and memory stall penalties.

**Reward formulation:**
\[
R_t = -\alpha L_t - \beta E_t - \gamma S_t + \delta U_t
\]
where:
- \(L_t\): latency term
- \(E_t\): energy term
- \(S_t\): memory stall penalty
- \(U_t\): system utilization efficiency

---
## License
MIT
