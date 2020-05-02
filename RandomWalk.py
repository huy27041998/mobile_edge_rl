import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
alphas = [0.05, 0.1, 0.15]
gamma = 1
nepisode = 100
N = 5
state = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
true_value = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
total_error = np.zeros(nepisode + 1)
runs = 100
draw_result = []
plt.subplot(122)
def get_reward(new_position):
    return 0
for alpha in alphas:
    for i in range(runs):
        storage = []
        for j in range(0, nepisode+1):
            terminated = False
            position = 3
            rms = np.linalg.norm(state[1:6] - true_value) / N
            storage.append(rms)
            if (alpha == 0.1 and j % 50 == 0 and i == 1):
                draw_result.append(state.copy()[1:N+1])
            while (not terminated):
                direction = np.random.choice([-1, 1])
                new_position = position + direction
                reward = get_reward(new_position)
                if new_position == 0 or new_position == len(state)-1:
                    terminated = True
                #update reward
                state[position] += alpha * (reward + gamma * state[new_position] - state[position])
                position = new_position
        total_error += storage
        state = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
    total_error /= runs
    plt.plot(total_error, label= 'alpha = %.02f' % (alpha))
plt.xlabel('episodes')
plt.ylabel('RMS')
plt.legend()
xaxis = ['A', 'B', 'C', 'D', 'E']
plt.subplot(121)
for i, d in enumerate(draw_result):
    plt.plot(xaxis, d, label = '%d episodes' %(50 * i))
plt.plot(xaxis, true_value, 'mx', label = 'True value')
plt.legend()
# plt.plot(nstate, true_value, nstate, state[1:6])
# plt.title('V_pi and T')
plt.show()