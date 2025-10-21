
import yaml, torch, numpy as np
from src.env import MultiAgentCPUMemEnv
from src.agents.maddpg import MADDPGAgent
from src.utils import set_seed, ensure_dir

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    env_cfg, agent_cfg = cfg["env"], cfg["agent"]
    set_seed(cfg.get("seed",0))
    env = MultiAgentCPUMemEnv(env_cfg["num_cpu_agents"], env_cfg["num_mem_agents"])
    n_agents = env.num_cpu + env.num_mem
    obs_dim = env.num_cpu + env.num_mem
    act_dim = 1

    agent = MADDPGAgent(obs_dim, act_dim, n_agents, agent_cfg)

    for ep in range(100):
        obs = env.reset()
        total_r = 0
        for t in range(env_cfg["episode_length"]):
            actions_cpu = [agent.act(obs["cpu"]) for _ in range(env.num_cpu)]
            actions_mem = [agent.act(obs["mem"]) for _ in range(env.num_mem)]
            obs, r, done, info = env.step(actions_cpu, actions_mem)
            total_r += r
            if done: break
        print(f"Episode {ep}: reward={total_r:.2f}")
