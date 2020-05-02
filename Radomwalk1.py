import numpy as np
np.set_printoptions(precision=3, suppress=True)

class Chain:
    gamma = 1       # Discount factor for future rewards (=1 because the game is finite and deterministic)

    def __init__(self, alpha = 0.1, values = np.ones(7)*0.5):
        self.nodes = ["Left", "A", "B", "C", "D", "E", "Right"]
        self.position = 3       # Initial position is "C"
        self.node = self.nodes[self.position]
        self.values = values
        self.terminated = False
        self.alpha = alpha

    def move(self):
        if not self.terminated:
            direction = np.random.choice(["left", "right"])

            if direction == "left":
                new_position = self.position - 1
            elif direction == "right":
                new_position = self.position + 1

            reward = self.get_reward(new_position)

            # self.update_value_table_immediate_reward(self.position, new_position)
            self.update_value_table_future_reward(self.position, new_position)

            self.position = new_position
            self.node = self.nodes[self.position]

            if (self.node == "Left") or (self.node == "Right"):
                self.terminated = True

        else:
            print("Moving is not possible as the random walk has already terminated.")

    def get_reward(self, new_position):
        return 1.0 if self.nodes[new_position] == "Right" else 0.0

    def update_value_table_immediate_reward(self, old_position, new_position):
        reward = self.get_reward(new_position)
        self.values[old_position] += self.alpha * (reward - self.values[old_position])

    def update_value_table_future_reward(self, old_position, new_position):
        if self.nodes[new_position] in ("Left","Right"):
            reward = self.get_reward(new_position)
        else:
            reward = self.get_reward(new_position) + self.gamma * self.values[new_position]
        self.values[old_position] += self.alpha * (reward - self.values[old_position])

value_estimates = np.ones(7) * 0.0      # Initialize estimates at all 0.5

N_episodes = 1000

for episode in range(N_episodes):
    c = Chain(values = value_estimates)         # Use value estimates from previous iteration
    while not c.terminated:
        c.move()
    value_estimates = c.values                  # Update value estimates

print(value_estimates[1:-1])