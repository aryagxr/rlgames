from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseEnvironment(ABC):
    """Base class for all environments in the RL framework"""
    
    def __init__(self, width: int, height: int, start: Tuple[int, int] = (0, 0), goal: Tuple[int, int] = (3, 3)):
        self.width = width
        self.height = height
        self.x = start[0]
        self.y = start[1]
        self.goal = goal
        self.set_actions_and_rewards()

    def get_state(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def get_actions(self) -> Dict[Tuple[int, int], List[str]]:
        return self.actions

    def get_rewards(self) -> Dict[Tuple[int, int], float]:
        return self.rewards

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def set_state(self, state: Tuple[int, int]) -> None:
        self.x = state[0]
        self.y = state[1]

    @abstractmethod
    def set_actions_and_rewards(self) -> None:
        pass

    @abstractmethod
    def take_action(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        pass

    def print_policy(self, policy: Dict[Tuple[int, int], str]) -> None:
        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        print("Policy π:")
        for i in range(self.height):
            row = ''
            for j in range(self.width):
                state = (i, j)
                if self.is_terminal(state):
                    row += ' G '  # Goal state
                else:
                    action = policy.get(state, ' ')
                    symbol = action_symbols.get(action, ' ')
                    row += f' {symbol} '
            print(row)

    def print_values(self, V: Dict[Tuple[int, int], float]) -> None:
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                state = (i, j)
                if self.is_terminal(state):
                    row += "  G   "
                else:
                    v = V.get(state, 0)
                    row += f"{v:5.2f} "
            print(row) 