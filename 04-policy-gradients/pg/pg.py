# this code is modified from pg-pong.py by karpathy

import gymnasium as gym
import numpy as np
import ale_py


# TODO:
# > preprocess frame
# > frame difference


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")



#hyperparameters
h = 200
batchsize = 10
lr = 1e-4
gamme = 0.9
decay = 0.99

num_a = env.action_space.n
dim = 80*80
w1 = np.random.randn(h,dim) / np.sqrt(dim)
w2 = np.random.randn(num_a,h) / np.sqrt(h)





def processframe(f):
    g = np.mean(f, axis=2)
    crop = g[35:195, :]
    ds = crop[::2,::2]
    norm = (ds > 0).astype(np.float32) # bg is 0, ball is 1
    norm = norm.ravel()
    return norm




obs, info = env.reset()
prevf = None
for _ in range(5):
    action = env.action_space.sample()
    obs, r, term, trnc, info = env.step(action)
    curf = processframe(obs)
    f = curf - prevf if prevf is not None else np.zeros(dim)
    prevf = curf
    
    
    
    if term or trnc:
        obs, info = env.reset()

env.close()


