

class Grid:
    def __init__(self, width, height, start=(0,0)):
        self.width = width
        self.height = height
        self.x = start[0]
        self.y = start[1]

    def get_state(self):
        return (self.x, self.y)



    def get_actions(self):
        return self.actions
    


    def get_rewards(self):
        return self.rewards


    def is_terminal(self, state):
        if state in self.actions:
            return False
        return True
    


    def set_rewards(self, rewards):
        self.rewards = rewards
        

    def set_actions(self, actions):
        self.actions = actions


    def set_states(self, state):
        self.x = state[0]
        self.y = state[1]




'''
# Grid World Example

# # # G
# # # #
# # # #
S # # # 

'''  

def grid_world():
    grid = Grid(4,4)
    actions = {
        (0, 0): ['right', 'down'],
        (0, 1): ['left', 'right', 'down'],
        (0, 2): ['left', 'right', 'down'],
        # (0, 3): ['left', 'down'],
        (1, 0): ['up', 'right', 'down'],
        (1, 1): ['up', 'left', 'right', 'down'],
        (1, 2): ['up', 'left', 'right', 'down'],
        (1, 3): ['up', 'left', 'down'],
        (2, 0): ['up', 'right', 'down'],
        (2, 1): ['up', 'left', 'right', 'down'],
        (2, 2): ['up', 'left', 'right', 'down'],
        (2, 3): ['up', 'left', 'down'],
        (3, 0): ['up', 'right'],
        (3, 1): ['up', 'left', 'right'],
        (3, 2): ['up', 'left', 'right'],
        (3, 3): ['up', 'left']
    }
    rewards = {(0, 3): 1, (1, 3): -1, (2, 3): -1, (3, 3): -1}
    grid.set_actions(actions)
    grid.set_rewards(rewards)
    return grid



