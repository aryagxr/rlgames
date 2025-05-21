from environments import Maze
import random
import numpy as np

class qagent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, eps=0.9):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q_table = {}
        self.init_q_table()

    def init_q_table(self):
        actions_dict = self.maze.get_actions()
        for s in actions_dict:
            for a in actions_dict[s]:
                self.q_table[(s, a)] = 0.0

    def select_action(self, s):
        if random.random() < self.eps: #explore
            return random.choice(self.maze.get_actions()[s])
        else: #exploit
            q_values = [self.q_table[(s, a)] for a in self.maze.get_actions()[s]]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.maze.get_actions()[s], q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_value(self, s, a, r, next_s, next_a):
        old_value = self.q_table[(s, a)]
        
        if self.maze.is_terminal(next_s):
            target = r
        else:
            target = r + self.gamma * self.q_table[(next_s, next_a)]
            
        self.q_table[(s, a)] = (1 - self.alpha) * old_value + self.alpha * target



    def train(self, num_episodes=1000, max_steps=100):
        for episode in range(num_episodes):
            self.maze.set_state((0, 0))
            s = self.maze.get_state()
            a = self.select_action(s)
            total_r = 0

            for step in range(max_steps):
                next_s = self.maze.take_action(s, a)
                r = self.maze.get_rewards()[next_s]

                if self.maze.is_terminal(next_s):
                    self.update_q_value(s, a, r, next_s, None)
                    break
                else:
                    a_next = self.select_action(next_s)
                    self.update_q_value(s, a, r, next_s, a_next)

                    s = next_s
                    a = a_next
                    total_r += r

            self.eps = max(0.01, self.eps * 0.995)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_r}")


    def get_policy(self):
        policy = {}
        for s in self.maze.get_actions():
            if not self.maze.is_terminal(s):
                q_values = [self.q_table[(s, a)] for a in self.maze.get_actions()[s]]
                best_a = self.maze.get_actions()[s][np.argmax(q_values)]
                policy[s] = best_a
        return policy

def main():
    maze = Maze(width=5, height=5, start=(0, 0), goal=(4, 4))
    agent = qagent(maze)
    
    print("Initial Maze Environment:")
    print(maze)
    
    print("\nTraining the agent...")
    agent.train(num_episodes=1000)
    
    print("\nLearned Policy:")
    policy = agent.get_policy()
    maze.print_policy(policy)
    
    print("\nTesting the learned policy:")
    maze.set_state((0, 0))
    s = maze.get_state()
    steps = 0
    max_steps = 100
    
    while not maze.is_terminal(s) and steps < max_steps:
        a = policy[s]
        s = maze.take_action(s, a)
        maze.set_state(s)
        steps += 1
        print(f"\nStep {steps}:")
        print(maze)

if __name__ == "__main__":
    main()

