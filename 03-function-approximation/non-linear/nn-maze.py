# vanilla DQN to solve maze

from environments import Maze
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class qnetwork(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(qnetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        

    def forward(self, x):
        return self.mlp(x)


class vanillaDQN:
    def __init__(self, maze, hidden_size=64, lrate=0.001, gamma=0.99, eps=1.0):
        self.maze = maze
        self.gamma = gamma
        self.eps = eps
        self.input_size = 2 # x, y
        self.output_size = 4 # up, down, left, right
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.network = qnetwork(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_network = qnetwork(self.input_size, hidden_size, self.output_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lrate)
        
        self.memory = deque(maxlen=10000) # replay buffer
        self.batch_size = 32
        self.a_idx = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.a_idx_rev = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        
        self.target_update = 10 

    def state_tensor(self, s):
        x, y = s
        norm_state = torch.FloatTensor([x/self.maze.width, y/self.maze.height]).to(self.device)
        return norm_state
    

    def best_action(self, s): # e-greedy
        if random.random() < self.eps:
            return random.choice(self.maze.get_actions()[s])
        else:
            with torch.no_grad():
                state_tensor = self.state_tensor(s)
                q_values = self.network(state_tensor) # outputs 4 q-values for the 4 actions
                max_id = q_values.argmax().item() # returns idx of the max q-value
                return self.a_idx_rev[max_id] # converts the idx to action
    
    
    def train(self, ep=1000):
        for e in range(ep):
            self.maze.set_state((0, 0))
            s = self.maze.get_state()
            r_total = 0
            while not self.maze.is_terminal(s):
                a = self.best_action(s)
                next_s = self.maze.take_action(s, a)
                r = self.maze.get_rewards()[next_s]
                self.memory.append((s, a, r, next_s))
                if len(self.memory) > self.batch_size:
                    self.train_step()
                s = next_s
                r_total += r
            
            # Update target network periodically
            if (e + 1) % self.target_update == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            
            self.eps = max(self.eps * 0.995, 0.01)
            if (e+1) % 10 == 0:
                print(f"Episode {e+1} finished with reward {r_total}")


    def train_step(self):
        batch = random.sample(self.memory, self.batch_size)
        s_batch, a_batch, r_batch, next_s_batch = zip(*batch)
        s_tensor = torch.stack([self.state_tensor(s) for s in s_batch])
        next_s_tensor = torch.stack([self.state_tensor(s) for s in next_s_batch])
        a_tensor = torch.tensor([self.a_idx[a] for a in a_batch], dtype=torch.long).to(self.device)
        r_tensor = torch.tensor(r_batch, dtype=torch.float).to(self.device)

        q_val = self.network(s_tensor)
        q_val = q_val.gather(1, a_tensor.unsqueeze(1))

        with torch.no_grad():
            next_qval = self.target_network(next_s_tensor).max(dim=1)[0]
            target_q = r_tensor + self.gamma * next_qval

        loss = nn.MSELoss()(q_val.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_policy(self):
        policy = {}
        for s in self.maze.get_actions():
            if not self.maze.is_terminal(s):
                with torch.no_grad():
                    s_tensor = self.state_tensor(s)
                    q_val = self.network(s_tensor)
                    a_idx = q_val.argmax().item()
                    policy[s] = self.a_idx_rev[a_idx]
        return policy
    

    

def test_agent(agent, maze, episodes=5):
    print("\nTesting the agent...")
    for episode in range(episodes):
        maze.set_state((0, 0))
        state = maze.get_state()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        print(f"\nEpisode {episode + 1}:")
        print(maze)
        
        while not maze.is_terminal(state) and steps < max_steps:
            action = agent.best_action(state)
            next_state = maze.take_action(state, action)
            reward = maze.get_rewards()[next_state]
            
            state = next_state
            maze.set_state(state)
            total_reward += reward
            steps += 1
            
            print(f"\nStep {steps}:")
            print(maze)
            print(f"Action: {action}, Reward: {reward}")
        
        print(f"\nEpisode finished in {steps} steps with total reward: {total_reward}")

if __name__ == "__main__":
    obstacles = [
        (0, 4),  
        (1, 0), (1, 1), (1, 4),
        (2, 2), (2, 4),
        (3, 1), (3, 2),
        (4, 0),
        (5, 2), (5, 5),
        (6, 1), (6, 4)
    ]
    
    maze = Maze(width=7, height=7, start=(0, 0), goal=(6, 6), obstacles=obstacles)
    print("\nInitial Maze Environment:")
    print(maze)
    
    agent = vanillaDQN(maze)
    print("\nTraining the agent...")
    agent.train(ep=1000)  
    
    from visual.visualize_maze import visualize_training
    visualize_training(maze, agent, episodes=3)
