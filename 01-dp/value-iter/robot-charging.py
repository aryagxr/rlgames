# simple golf example
import numpy as np
import time

gamma = 0.9 # discount factor
theta = 0.01 #convergence threshold

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
        "search for charger": {s1: 0.8, s0: 0.2}
    },
    s1: {
        "docking for charge": {s2: 0.8, s1: 0.2},
        "stop moving": {s0: 0.8, s1: 0.2}
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


# value iteration
def value_iter():
    iter = 0
    converged = False
    while not converged:
        print(f"\n=== Iteration {iter} ===")
        print(f"{'State':<10} {'Old V':>10} {'New V':>10} {'Delta':>10}")
        delta = 0
        for s in states:
            val = value_table[s]
            cur_action = []
            for a in actions[s]:
                a_val = 0
                for next_state in trans_prob[s][a]:
                    a_val += trans_prob[s][a][next_state] * (rewards[next_state] + gamma * value_table[next_state])
                cur_action.append(a_val)
            if cur_action:
                value_table[s] = max(cur_action)
            else:
                value_table[s] = 0
            delta = max(delta, abs(val - value_table[s]))
            print(f"{s:<10} {val:>10.4f} {value_table[s]:>10.4f} {abs(val - value_table[s]):>10.4f}")

        if delta < theta:
            converged = True
        print(f"Total Delta: {delta:.6f}")
        iter += 1   


    # extract policy
    policy = {}
    for s in states:
        if not actions[s]:
            policy[s] = None
            continue
        action_vals = {}
        for a in actions[s]:
            a_val = 0
            for next_state in trans_prob[s][a]:
                a_val += trans_prob[s][a][next_state] * (rewards[next_state] + gamma * value_table[next_state])
            action_vals[a] = a_val
        best_action = max(action_vals, key=action_vals.get)
        policy[s] = best_action
    
    print("\n=== Final Results ===")
    print("Value Table:")
    print(value_table)
    print("Policy:")
    print(policy)
        

        

value_iter()