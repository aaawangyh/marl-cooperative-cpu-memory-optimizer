
from src.env import MultiAgentCPUMemEnv
from src.agents.maddpg import MADDPGAgent
import numpy as np

def main():
    env = MultiAgentCPUMemEnv()
    agent = MADDPGAgent(obs_dim=6, act_dim=1, n_agents=6, cfg={})
    R = []
    for ep in range(10):
        obs = env.reset()
        tot = 0
        for _ in range(200):
            actions_cpu = [agent.act(obs["cpu"]) for _ in range(env.num_cpu)]
            actions_mem = [agent.act(obs["mem"]) for _ in range(env.num_mem)]
            obs, r, d, i = env.step(actions_cpu, actions_mem)
            tot += r
            if d: break
        R.append(tot)
    print("Average Reward:", np.mean(R))

if __name__ == "__main__":
    main()
