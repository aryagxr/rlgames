# simple golf example
import numpy as np
import time

gamma = 0.9 #discount factor
theta = 0.01

s0 = "IDLE"
s1 = "MOVING"
s2 = "CHARGING"

states = [s0, s1, s2]
actions = {
    s0: ["search for charger"],
    s1: ["docking for charge", "stop moving"], 
    s2: []
}

# transition probabilities
# T(s' | s, a)
trans_prob = {
    # from state s0, when performing action "search for charger",
    # the chances of moving (s1) is 0.9 & chances of not is 0.1 
    s0: {
        "search for charger": {s1: 0.9, s0: 0.1}
    },
    s1: {
        "docking for charge": {s2: 0.9, s1: 0.1},
        "stop moving": {s0: 0.9, s1: 0.1}
    }
}


value_table = {
    s0: 0.0,
    s1: 0.0,
    s2: 0.0
}

rewards = {
    s0: 0.0,
    s1: 0.0,
    s2: 10.0
}  

def reached_terminal_state(state):
    return state == s2




# prob = trans_prob[s0]["stop moving"][s1]
# print(prob)


# value iteration
def value_iter():
    converged = False
    while not converged:
        delta = 0
        for s in states:
            print(f"state: {s}")
            val = value_table[s]
            print(f"val: {val}")
            cur_action = []
            for a in actions[s]:
                a_val = 0
                print(f"action: {a}")
                for next_state in trans_prob[s][a]:
                    print(f"next_state: {next_state}")
                    a_val += trans_prob[s][a][next_state] * (rewards[next_state] + gamma * value_table[next_state])
                cur_action.append(a_val)
                print(f"cur_action: {cur_action}")
            if cur_action:
                value_table[s] = max(cur_action)
                print(f"max: {value_table[s]}")
                print(value_table)
            else:
                value_table[s] = 0
            delta = max(delta, abs(val - value_table[s]))
            print(f"delta: {delta}")
        if delta < theta:
            converged = True

    # extract policy
    # policy = {}
    # for s in states:
        
            


value_iter()