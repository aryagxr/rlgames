from typing import Dict, List, Tuple
from .base import BaseEnvironment

class Maze(BaseEnvironment):
    """A maze environment with obstacles for reinforcement learning"""
    
    def __init__(
        self,
        width: int,
        height: int,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (3, 3),
        obstacles: List[Tuple[int, int]] = None
    ):
        if obstacles is None:
            self.obstacles = [(1, 1), (1, 2), (2, 1), (2, 2)]
        else:
            self.obstacles = obstacles
        super().__init__(width, height, start, goal)

    def is_obstacle(self, state: Tuple[int, int]) -> bool:
        return state in self.obstacles

    def set_actions_and_rewards(self) -> None:
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

    def take_action(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        x, y = state
        next_state = {
            'up': (max(x - 1, 0), y),
            'down': (min(x + 1, self.height - 1), y),
            'left': (x, max(y - 1, 0)),
            'right': (x, min(y + 1, self.width - 1))
        }
        new_state = next_state[action]
        # If the new state is an obstacle, stay in the current state
        return state if self.is_obstacle(new_state) else new_state

    def __str__(self) -> str:
        result = ""
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state == (self.x, self.y):
                    result += " 👾 "
                elif state == self.goal:
                    result += " G "
                elif self.is_obstacle(state):
                    result += " # "
                else:
                    result += " . "
            result += "\n"
        return result

def create_maze() -> Maze:
    maze = Maze(width=4, height=4, start=(0, 0), goal=(3, 3))
    print("\nMaze Environment:")
    print(maze)
    return maze 