# Linear funcition approximation

from environments import Maze
import numpy as np
import time
import os


class LinearFunctionApproximator:
    def __init__(self, env, alpha=0.01, gamma=0.9, eps=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # weights
        self.theta = np.random.randn(self.get_num_features())
    


    def get_num_features(self):
        # 2: position; 2: goal relative; 4: actions
        return 8
    
    def phi(self, s, a):
        # φ(s, a)
        # s is a tuple (x, y)
        # a is a string
        features = np.zeros(self.get_num_features())
        x, y = s
        gx, gy = self.env.goal
        features[0] = x / self.env.width
        features[1] = y / self.env.height
        features[2] = (gx - x) / self.env.width
        features[3] = (gy - y) / self.env.height
        a_id = {'up': 4, 'down': 5, 'left': 6, 'right': 7}
        features[a_id[a]] = 1
        return features
    
    # Q(s, a) = θ^T φ(s, a)
    def q(self, s, a):
        phi = self.phi(s, a)
        return np.dot(self.theta, phi)
    

    def best_action(self, s):
        if np.random.rand() < self.eps: # explore
            return np.random.choice(self.env.get_actions()[s])
        else: # exploit
            best_action = None
            best_q = -float('inf')
            for a in self.env.get_actions()[s]:
                q = self.q(s, a)
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action
        

    # error = r + γ * max_a Q(s', a) - Q(s, a)
    # Update = α * error * φ(s, a)
    def update(self, s, a, r, s_next):
        q_pred = self.q(s, a) # Q(s, a)
        next_q = 0 # max_a Q(s', a)
        if not self.env.is_terminal(s_next):
            a_next = self.env.get_actions()[s_next]
            next_q = max(self.q(s_next, a) for a in self.env.get_actions()[s_next])

        error = r + self.gamma * next_q - q_pred
        features = self.phi(s, a)
        self.theta += self.alpha * error * features


    
    def train(self, episodes=100):
        for e in range(episodes):
            s = self.env.get_state()
            r_total = 0
            while not self.env.is_terminal(s):
                a = self.best_action(s)
                # take action a
                s_next = self.env.take_action(s, a)
                r = self.env.get_rewards()[s_next]
                self.update(s, a, r, s_next)
                s = s_next
                r_total += r
            self.eps = max(0.01, self.eps * 0.995)
            # progress print
            if (e + 1) % 10 == 0:
                print(f"Episode {e + 1}, Total Reward: {r_total}")


    # for each state, return action with highest q-value
    # calculating qvalue on spot using learned weights
    def get_policy(self):
        policy = {}
        for s in self.env.get_actions():
            if not self.env.is_terminal(s):
                best_a = None
                best_q = -float('inf')
                for a in self.env.get_actions()[s]:
                    q_val = self.q(s, a)
                    if q_val > best_q:
                        best_q = q_val
                        best_a = a
                policy[s] = best_a
        return policy
    

    def test(self, policy, steps=100):
        s = self.env.get_state()
        total_reward = 0
        path = [s]
        
        for _ in range(steps):
            if self.env.is_terminal(s):
                break
                
            a = policy[s]
            s_next = self.env.take_action(s, a)
            r = self.env.get_rewards()[s_next]
            s = s_next
            total_reward += r
            path.append(s)
            print(f"State: {s}, Action: {a}, Reward: {r}")
            
        print(f"\nTest Results:")
        print(f"Total Reward: {total_reward}")
        print(f"Path Length: {len(path)}")
        print(f"Final State: {s}")
        print(f"Reached Goal: {self.env.is_terminal(s)}")
        
        return total_reward, path
    
    
    def animate_terminal(self, path, policy, delay=0.5):
        for state in path:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            maze_str = []
            for y in range(self.env.height):
                row = []
                for x in range(self.env.width):
                    if (y, x) == state: 
                        row.append('👾')  
                    elif (y, x) == self.env.goal:  
                        row.append('G')  
                    elif (y, x) in self.env.obstacles:  
                        row.append('#') 
                    else:
                        row.append('.')
                maze_str.append(' '.join(row))
            
           
            print('\n'.join(maze_str))

            if not self.env.is_terminal(state):
                print(f"\nState: {state}")
                print(f"Action: {policy[state]}")
            else:
                print("\nGoal Reached!")
            
            time.sleep(delay)


'''
 S  .  .  #  .  .  . 
 #  #  .  .  #  .  . 
 .  .  #  .  #  .  . 
 .  #  #  .  .  .  . 
 #  .  .  .  .  .  . 
 .  .  #  .  .  #  . 
 .  #  .  .  #  .  G 
'''


def main():
    obstacles = [
        (0, 3),
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
    
    lfa = LinearFunctionApproximator(maze, alpha=0.01, gamma=0.9, eps=0.1)
    print("\nTraining the agent...")
    lfa.train(episodes=1000)
    
    print("\nLearned Policy:")
    policy = lfa.get_policy()
    maze.print_policy(policy)
    
    print("\nTesting the learned policy:")
    maze.set_state((0, 0))  
    total_reward, path = lfa.test(policy)
    
    print("\nAnimating path in terminal...")
    lfa.animate_terminal(path, policy)
    
    print("\nFinal Path:")
    for state in path:
        maze.set_state(state)
        print(maze)
        print(f"State: {state}")
        if maze.is_terminal(state):
            break
        print("Action:", policy[state])
        print("-" * 20)

if __name__ == "__main__":
    main()
        

    
    
    

