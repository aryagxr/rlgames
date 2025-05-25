from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
import time
from environments import Maze

def create_maze_panel(maze):
    return Panel(maze.__str__(), title="Maze State", border_style="blue")

def create_reward_panel(maze, state):
    table = Table(title="Rewards")
    table.add_column("State")
    table.add_column("Reward")
    table.add_row(str(state), str(maze.get_rewards()[state]))
    for action in maze.get_actions()[state]:
        next_state = maze.take_action(state, action)
        table.add_row(f"{state} → {next_state}", str(maze.get_rewards()[next_state]))
    return Panel(table, title="Rewards", border_style="green")

def create_policy_panel(policy, state):
    table = Table(title="Policy")
    table.add_column("State")
    table.add_column("Action")
    table.add_row(str(state), policy.get(state, "N/A"))
    return Panel(table, title="Learned Policy", border_style="yellow")

def create_policy_arrows_panel(maze, policy):
    grid = []
    for i in range(maze.height):
        row = []
        for j in range(maze.width):
            state = (i, j)
            if state in policy:
                action = policy[state]
                arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}.get(action, '?')
                row.append(arrow)
            elif state == maze.goal:
                row.append('G')
            elif state in maze.obstacles:
                row.append('#')
            else:
                row.append('.')
        grid.append(' '.join(row))
    return Panel('\n'.join(grid), title="Policy Arrows", border_style="magenta")

def build_layout(maze, agent, state):
    layout = Layout()

    layout.split_column(
        Layout(name="row1"),
        Layout(name="row2")
    )

    layout["row1"].split_row(
        Layout(name="maze"),
        Layout(name="rewards")
    )
    layout["row2"].split_row(
        Layout(name="policy"),
        Layout(name="policy_arrows")
    )

    layout["maze"].update(create_maze_panel(maze))
    layout["rewards"].update(create_reward_panel(maze, state))
    layout["policy"].update(create_policy_panel(agent.get_policy(), state))
    layout["policy_arrows"].update(create_policy_arrows_panel(maze, agent.get_policy()))

    return layout

def visualize_training(maze, agent, episodes=3, delay=0.5):
    console = Console()

    with Live(console=console, screen=True, auto_refresh=False) as live:
        for episode in range(episodes):
            maze.set_state((0, 0))
            state = maze.get_state()
            total_reward = 0
            steps = 0
            max_steps = 100

            while not maze.is_terminal(state) and steps < max_steps:
                action = agent.best_action(state)
                next_state = maze.take_action(state, action)
                reward = maze.get_rewards()[next_state]

                layout = build_layout(maze, agent, state)
                live.update(layout, refresh=True)

                state = next_state
                maze.set_state(state)
                total_reward += reward
                steps += 1
                time.sleep(delay)
