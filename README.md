# RL Games

A collection of reinforcement learning algorithms and environments implemented in Python.

## Getting Started

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/aryagxr/rlgames.git
cd rlgames
```

### 2. Create a Virtual Environment

You can use either `venv` (Python's built-in virtual environment) or `conda` (Anaconda/Miniconda):

#### Option 1: Using venv (Python's built-in virtual environment)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

#### Option 2: Using conda (Anaconda/Miniconda)

```bash
# Create a conda environment
conda create -n rlgames

# Activate the conda environment
conda activate rlgames
```

### 3. Install Dependencies

Install the required packages:

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 4. Run the Examples

You can now run any of the examples:

```bash
# Run Q-Learning example
python 02-td-learning/q-learning/q-maze.py

# Run SARSA example
python 02-td-learning/sarsa/sarsa-maze.py
```


## Quick Example

```python
from environments import Maze
from q_learning import qagent

# Create a maze environment
maze = Maze(width=5, height=5, start=(0, 0), goal=(4, 4))

# Create and train a Q-learning agent
agent = qagent(maze)
agent.train(num_episodes=1000)

# Get and display the learned policy
policy = agent.get_policy()
maze.print_policy(policy)
```

## Requirements

- Python >= 3.6
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 