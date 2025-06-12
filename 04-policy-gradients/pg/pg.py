# this code is modified from pg-pong.py by karpathy

import gymnasium as gym
import numpy as np
import ale_py
import pickle
import wandb


#hyperparams
h = 200
batchsize = 10
lr = 1e-5
gamma = 0.99
decay = 0.99


wandb.init(
    project="breakout-pg",
    config={
        "learning_rate": lr,
        "batch_size": batchsize,
        "gamma": gamma,
        "decay": decay,
        "hidden_size": h
    }
)

resume = False
if resume:
    with open("pg_breakout.pkl", "rb") as f:
        w1, w2, rmsprop = pickle.load(f)


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")



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
    dw2 = np.dot(eplogp.T, eph)
    dh1 = np.dot(eplogp, w2)
    dh1[eph <= 0] = 0
    dw1 = np.dot(epx.T, dh1)
    dw1 = dw1.T
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
max_ep = 1000
grad_buffer = {'dw1': np.zeros_like(w1), 'dw2': np.zeros_like(w2)}
rmsprop = {'dw1': np.zeros_like(w1), 'dw2': np.zeros_like(w2)}
batch_rewards = []


while True:
    if ep >= max_ep:
        break
    print(f"\nEpisode {ep}")
    curf = processframe(obs)
    f = curf - prevf if prevf is not None else np.zeros(dim)
    prevf = curf
    
    aprob, h = forward(f)
    action = np.random.choice(len(aprob), p=aprob)
    xs.append(f)
    hs.append(h)
    actions.append(action)

    
    obs, r, term, trnc, info = env.step(action)
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
        
        for h, a, r in zip(eph, epa, dr):
            logits = np.dot(w2, h)
            p = softmax(logits)
            dlogits = p.copy()
            dlogits[a] -= 1
            dlogits *= r
            eplogp.append(dlogits)
        
        eplogp = np.array(eplogp)

        grads = backward(eph, eplogp, epx)

        grad_buffer['dw1'] += grads['dw1']
        grad_buffer['dw2'] += grads['dw2']

        if ep % batchsize == 0:
            batch_avg_reward = np.mean(batch_rewards) if batch_rewards else 0
            print(f"Batch {ep//batchsize} avg reward: {batch_avg_reward:.2f}")
            batch_rewards = []
            
            for k in rmsprop:
                g = grad_buffer[k]
                rmsprop[k] = decay * rmsprop[k] + (1 - decay) * g**2
                update = lr * g / (np.sqrt(rmsprop[k]) + 1e-5)

                if k == 'dw1':
                    w1 += update
                else:
                    w2 += update

                grad_buffer[k] = np.zeros_like(g)

        avg_reward = np.sum(ep_rewards)
        running_reward = avg_reward if running_reward == 0 else running_reward * 0.99 + avg_reward * 0.01
        batch_rewards.append(avg_reward)
        print(f"Episode {ep} reward: {avg_reward:.1f}, running mean: {running_reward:.1f}")

        wandb.log({
            "episode": ep,
            "total_reward": avg_reward,
            "running_reward": running_reward,
            "mean_discounted_reward": np.mean(dr),
            "w1_norm": np.linalg.norm(w1),
            "w2_norm": np.linalg.norm(w2),
            "batch_avg_reward": batch_avg_reward if ep % batchsize == 0 else None
        })

        if ep % 5 == 0:
            with open("pg_breakout.pkl", "wb") as f:
                pickle.dump([w1, w2, rmsprop], f)

            wandb.save("pg_breakout.pkl")

        obs, info = env.reset()
        prevf = None
        xs, hs, actions = [], [], []
        ep_rewards = []
        running_reward = 0
        

env.close()


