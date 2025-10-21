import os, random, numpy as np
def set_seed(s): random.seed(s); np.random.seed(s)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
