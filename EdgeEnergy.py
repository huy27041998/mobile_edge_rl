from EdgeEnergyEnv import EdgeEnergyEnv
import numpy as np

def readFile(filename):
    gs = np.empty((0, 96), float)
    with open(filename, 'r') as f:
        days = f.read().splitlines()
        for day in days:
            day_g = np.array(day.split())
            day_g = day_g.astype(np.float)
            day_g = np.reshape(day_g, (1, 96))
            gs = np.append(gs, np.array(day_g), axis=0)
    return gs
gs = readFile('data1.txt')
s = EdgeEnergyEnv('qlearning')
for g in gs:
    s.g = g
    while(True):
        s.update_bt()
        new_cost, best_action = s.calculate_action()
        s.update_state(new_cost, best_action)
        if s.t == 0:
            break
print(s.cost)