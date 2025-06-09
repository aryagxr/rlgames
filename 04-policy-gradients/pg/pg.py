# this code is modified from pg-pong.py by karpathy

import gymnasium as gym
import numpy as np
import ale_py


# TODO:
# > preprocess frame
# > frame difference
# > forwards policy network
# > sample action
# > storing reward trajectories
# > discounting rewards
# > compute gradients

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")



#hyperparameters
h = 200
batchsize = 10
lr = 1e-4
gamma = 0.9
decay = 0.99

num_a = env.action_space.n
dim = 80*80
w1 = np.random.randn(h,dim) / np.sqrt(dim) #6400x200
w2 = np.random.randn(num_a,h) / np.sqrt(h) #200*4


def processframe(f):
    g = np.mean(f, axis=2)
    crop = g[35:195, :]
    ds = crop[::2,::2]
    norm = (ds > 0).astype(np.float32) # bg is 0, ball is 1
    norm = norm.ravel()
    return norm

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def forward(x):
    h1 = np.dot(w1,x)
    h1[h1<0] = 0
    logits = np.dot(w2, h1)
    p = softmax(logits)
    return p, h1


def discount(r):
    disc_r = np.zeros_like(r)
    Gt = 0 #Gt = rt + gamma*rt+1 + gamma^2*rt+2 + ...
    for i in reversed(range(len(r))):
        Gt = r[i] + gamma * Gt
        disc_r[i] = Gt
    return disc_r


def backward(eph, eplogp, epx):
    # dw1 = np.zeros_like(w1)
    # dw2 = np.zeros_like(w2)
    dw2 = np.dot(eplogp.T, eph)
    dh1 = np.dot(eplogp, w2)
    dh1[eph < 0] = 0
    dw1 = np.dot(dh1.T, epx)
    print("dw1 shape:", dw1.shape)
    print("dw2 shape:", dw2.shape)
    return {
        "dw1": dw1,
        "dw2": dw2
    }
    
    





obs, info = env.reset()
prevf = None
xs, hs, actions = [], [], []
ep_rewards = []
running_reward = 0
ep = 0
max_ep = 10


while True:
    if ep >= max_ep:
        break
    print(f"\nEpisode {ep}")
    curf = processframe(obs)
    f = curf - prevf if prevf is not None else np.zeros(dim)
    prevf = curf
    
    aprob, h = forward(f)
    action = np.random.choice(len(aprob), p=aprob) #weighted
    xs.append(f)
    hs.append(h)
    actions.append(action)
    # print("Sampled probs:", aprob)
    # print("Chosen action:", action)
    
    obs, r, term, trnc, info = env.step(action)
    print(f"Reward: {r}, Terminated: {term}, Truncated: {trnc}")
    ep_rewards.append(r)
    running_reward += r

    if term or trnc:
        ep += 1

        epx = np.array(xs) #steps * 6400
        eph = np.array(hs) #steps * 200
        # all the actions taken in this ep
        epa = np.array(actions) #steps * 1
        # all the rewards in this ep
        epr = np.array(ep_rewards) #steps * 1
        eplogp = []

        dr = discount(epr)
        dr = (dr - dr.mean()) / (dr.std() + 1e-8)
        print("\nEpisode Summary:")
        print("Raw rewards:", ep_rewards)
        print("Discounted rewards:", dr)
        print(f"Total episode reward: {running_reward}")
        print(f"Average discounted reward: {np.mean(dr)}")
        
        for h, a, r in zip(eph, epa, dr):
            logits = np.dot(w2, h)
            p = softmax(logits)
            dlogits = p.copy()
            dlogits[a] -= 1
            eplogp.append(dlogits)
        
        eplogp = np.array(eplogp)
        print("eplogp shape:", eplogp.shape)
        print("epx shape:", epx.shape)

        grads = backward(eph, eplogp, epx)
        print("grads:", grads)

        # Reset for next episode
        obs, info = env.reset()
        prevf = None
        xs, hs, actions = [], [], []
        ep_rewards = []
        running_reward = 0
        

env.close()


