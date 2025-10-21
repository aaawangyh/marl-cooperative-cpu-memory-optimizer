
import numpy as np

class MultiAgentCPUMemEnv:
    def __init__(self, num_cpu_agents=4, num_mem_agents=2, episode_length=1000, seed=0):
        self.rng = np.random.default_rng(seed)
        self.num_cpu = num_cpu_agents
        self.num_mem = num_mem_agents
        self.T = episode_length
        self.t = 0
        self.reset()

    def reset(self):
        self.t = 0
        self.cpu_load = np.zeros(self.num_cpu)
        self.mem_load = np.zeros(self.num_mem)
        return self._get_obs()

    def step(self, actions_cpu, actions_mem):
        # CPU agents decide compute allocation, mem agents allocate bandwidth
        self.cpu_load += np.array(actions_cpu) * self.rng.uniform(0.8, 1.2, len(actions_cpu))
        self.mem_load += np.array(actions_mem) * self.rng.uniform(0.8, 1.2, len(actions_mem))

        latency = np.mean(self.cpu_load) + np.mean(self.mem_load) * 0.5
        energy = np.mean(np.square(actions_cpu)) * 0.01 + np.mean(np.square(actions_mem)) * 0.02
        stalls = np.clip(np.mean(self.mem_load) - 1.0, 0, None)

        reward = -latency - 2.0 * energy - 0.5 * stalls
        self.t += 1
        done = self.t >= self.T
        obs = self._get_obs()
        return obs, reward, done, dict(latency=latency, energy=energy, stalls=stalls)

    def _get_obs(self):
        return dict(cpu=self.cpu_load.copy(), mem=self.mem_load.copy())
