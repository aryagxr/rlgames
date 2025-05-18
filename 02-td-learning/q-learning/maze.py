import random

class Maze:
    def __init__(self, width, height, start=(0, 0), goal=(3, 3)):
        self.width = width
        self.height = height
        self.x = start[0]
        self.y = start[1]
        self.goal = goal
        # can change the obstacles here
        self.obstacles = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.set_actions_and_rewards()

    def get_state(self):
        return (self.x, self.y)

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards

    def is_terminal(self, state):
        return state == self.goal

    def is_obstacle(self, state):
        return state in self.obstacles

    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]

    def set_actions_and_rewards(self):
        actions = {}
        rewards = {}

        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                rewards[state] = -1
                possible_actions = []
                
                if not self.is_obstacle(state):
                    if i > 0 and not self.is_obstacle((i-1, j)):
                        possible_actions.append('up')
                    if i < self.height - 1 and not self.is_obstacle((i+1, j)):
                        possible_actions.append('down')
                    if j > 0 and not self.is_obstacle((i, j-1)):
                        possible_actions.append('left')
                    if j < self.width - 1 and not self.is_obstacle((i, j+1)):
                        possible_actions.append('right')
                    
                    if state != self.goal:
                        actions[state] = possible_actions

        rewards[self.goal] = 10
        self.actions = actions
        self.rewards = rewards

    def __str__(self):
        result = ""
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state == (self.x, self.y):
                    result += " 🦕 "
                elif state == self.goal:
                    result += " G "
                elif self.is_obstacle(state):
                    result += " # "
                else:
                    result += " . "
            result += "\n"
        return result
    
    def generate_random_policy(maze, goal):
        height = maze.height
        width = maze.width
        policy = {}

        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }

        for x in range(height):
            for y in range(width):
                state = (x, y)
                if state == goal:
                    continue

                valid_actions = []
                if x > 0:
                    valid_actions.append('up')
                if x < height - 1:
                    valid_actions.append('down')
                if y > 0:
                    valid_actions.append('left')
                if y < width - 1:
                    valid_actions.append('right')

                policy[state] = random.choice(valid_actions)

        # print the policy with arrows
        print("Initial Random Policy π:")
        for i in range(height):
            row = ''
            for j in range(width):
                state = (i, j)
                if state == goal:
                    row += ' G '
                else:
                    action = policy.get(state, ' ')
                    symbol = action_symbols.get(action, ' ')
                    row += f' {symbol} '
            print(row)

        return policy
    

    def take_action(self, state, action):
        x, y = state
        next_state = {
            'up': (max(x - 1, 0), y),
            'down': (min(x + 1, self.height - 1), y),
            'left': (x, max(y - 1, 0)),
            'right': (x, min(y + 1, self.width - 1))
        }
        return next_state[action]
        


    def print_policy(self, policy, maze):
        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        print("Policy π:")
        for i in range(maze.height):
            row = ''
            for j in range(maze.width):
                state = (i, j)
                if maze.is_terminal(state):
                    row += ' G '  # Goal state
                else:
                    action = policy.get(state, ' ')
                    symbol = action_symbols.get(action, ' ')
                    row += f' {symbol} '
            print(row)


    



def create_maze():
    maze = Maze(width=4, height=4, start=(0, 0), goal=(3, 3))
    print("\nMaze Environment:")
    print(maze)
    return maze


def print_values(V, maze):
    for i in range(maze.height):
        row = ""
        for j in range(maze.width):
            state = (i, j)
            if maze.is_terminal(state):
                row += "  G   "
            else:
                v = V.get(state, 0)
                row += f"{v:5.2f} "
        print(row)



