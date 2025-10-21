
# Methodology

We model cooperative CPU–Memory optimization as a Dec-POMDP.
Each agent (CPU core or memory unit) observes its local queue length, utilization, and temperature.
Agents share a differentiable critic or mixing network (MADDPG or QMIX) to coordinate.

## Environment Features
- Dynamic task arrivals with variable compute/memory intensity.
- Shared global reward to align local objectives.
- Differentiable simulation with numpy-based backend for fast rollouts.

## Agents
- **MADDPG:** centralized critic with decentralized actors.
- **QMIX:** monotonic mixing network that ensures global consistency.

## Evaluation Metrics
- Average response time
- Energy efficiency (tasks per Joule)
- Memory stall rate
- Load balance across agents
