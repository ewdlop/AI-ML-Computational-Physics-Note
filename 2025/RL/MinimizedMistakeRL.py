import gym
import numpy as np

# Define RL Environment
env = gym.make('CartPole-v1')  # Example: Balancing a pole

# Initialize Policy
state = env.reset()
total_reward = 0

for _ in range(1000):  # Steps in an episode
    action = env.action_space.sample()  # Choose random action
    next_state, reward, done, _ = env.step(action)

    # Custom Reward Function: Least Mistakes = More Reward
    reward = max(0, reward - abs(next_state[2]))  # Penalize angle deviations

    total_reward += reward
    if done:
        break

print(f"Total Reward (Least Mistakes Optimization): {total_reward}")
