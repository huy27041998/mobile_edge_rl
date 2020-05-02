
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

WORLD_HEIGHT = 4
WORLD_WIDTH = 12
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

START = [3, 0]
GOAL = [3, 11]
def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards

def sarsa(q_value, expected=False, step_size=ALPHA):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards

def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

step_sizes = np.arange(0.1, 1.1, 0.1)
episodes = 1000
runs = 10
ASY_SARSA = 0
ASY_EXPECTED_SARSA = 1
ASY_QLEARNING = 2
INT_SARSA = 3
INT_EXPECTED_SARSA = 4
INT_QLEARNING = 5
methods = range(0, 6)

performace = np.zeros((6, len(step_sizes)))
for run in range(runs):
    for ind, step_size in list(zip(range(0, len(step_sizes)), step_sizes)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_expected_sarsa = np.copy(q_sarsa)
        q_q_learning = np.copy(q_sarsa)
        for ep in range(episodes):
            sarsa_reward = sarsa(q_sarsa, expected=False, step_size=step_size)
            expected_sarsa_reward = sarsa(q_expected_sarsa, expected=True, step_size=step_size)
            q_learning_reward = q_learning(q_q_learning, step_size=step_size)
            performace[ASY_SARSA, ind] += sarsa_reward
            performace[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
            performace[ASY_QLEARNING, ind] += q_learning_reward

            if ep < 100:
                performace[INT_SARSA, ind] += sarsa_reward
                performace[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                performace[INT_QLEARNING, ind] += q_learning_reward

performace[:3, :] /= episodes * runs
performace[3:, :] /= 100 * runs
labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
            'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

for method, label in zip(methods, labels):
    plt.plot(step_sizes, performace[method, :], label=label)
plt.xlabel('alpha')
plt.ylabel('reward per episode')
plt.legend()

plt.show()
plt.close()