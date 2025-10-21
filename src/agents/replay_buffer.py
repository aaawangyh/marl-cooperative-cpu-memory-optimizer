
from collections import deque, namedtuple
import random, numpy as np

Transition = namedtuple("Transition", "obs act rew next_obs done")

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))
    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        obs, act, rew, next_obs, done = map(np.stack, zip(*batch))
        return obs, act, rew, next_obs, done
    def __len__(self): return len(self.buf)
