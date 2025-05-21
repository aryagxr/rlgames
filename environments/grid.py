from typing import Dict, List, Tuple
from .base import BaseEnvironment

class Grid(BaseEnvironment):
    """A simple grid world environment for reinforcement learning"""
    
    def set_actions_and_rewards(self) -> None:
        actions = {}
        rewards = {}

        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                rewards[state] = -1
                possible_actions = []
                if i > 0:
                    possible_actions.append('up')
                if i < self.height - 1:
                    possible_actions.append('down')
                if j > 0:
                    possible_actions.append('left')
                if j < self.width - 1:
                    possible_actions.append('right')
                if state != self.goal:
                    actions[state] = possible_actions

        rewards[self.goal] = 10
        self.actions = actions
        self.rewards = rewards

    def take_action(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        x, y = state
        next_state = {
            'up': (max(x - 1, 0), y),
            'down': (min(x + 1, self.height - 1), y),
            'left': (x, max(y - 1, 0)),
            'right': (x, min(y + 1, self.width - 1))
        }
        return next_state[action]

    def __str__(self) -> str:
        result = ""
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state == (self.x, self.y):
                    result += " S "
                elif state == self.goal:
                    result += " G "
                else:
                    result += " . "
            result += "\n"
        return result

def create_grid_world() -> Grid:
    return Grid(width=4, height=4, start=(0, 0), goal=(3, 3)) 