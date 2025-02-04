import numpy as np
import random

class DrivingAI:
    def __init__(self):
        self.q_table = {}  # Q-learning table
        self.bad_action_memory = {}  # Tracks bad habits
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration vs exploitation

    def choose_action(self, state, possible_actions):
        """Choose an action using RL with some randomness (exploration)"""
        if random.uniform(0, 1) < self.epsilon:  # Explore
            return random.choice(possible_actions)
        return self.best_action(state, possible_actions)  # Exploit learned behavior

    def best_action(self, state, possible_actions):
        """Choose the action with the highest Q-value"""
        if state not in self.q_table:
            return random.choice(possible_actions)  # Default random choice
        return max(possible_actions, key=lambda a: self.q_table[state].get(a, 0))

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-values using RL"""
        if state not in self.q_table:
            self.q_table[state] = {}

        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0)
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state].get(action, 0) + \
                                      self.learning_rate * (reward + self.discount_factor * max_future_q)

    def track_bad_action(self, action, penalty):
        """Track negative reinforcement (bad habits)"""
        if penalty < 0:
            self.bad_action_memory[action] = self.bad_action_memory.get(action, 0) + 1

    def reverse_learn(self):
        """Unlearn bad habits if they occur too frequently"""
        for action, penalty in self.bad_action_memory.items():
            if penalty > 5:  # If action leads to bad outcomes frequently
                print(f"🚨 Unlearning bad action: {action}")
                self.bad_action_memory[action] = max(0, penalty - 1)  # Decay bad habit

    def decide_reinforce_or_unlearn(self, action, reward):
        """Tie-breaker logic: Decide whether to reinforce or unlearn"""
        if reward >= 0:
            print(f"✅ Reinforcing good action: {action}")
        else:
            print(f"❌ Unlearning bad habit: {action}")
            self.track_bad_action(action, reward)
            self.reverse_learn()

# Example: Autonomous Driving Scenario
ai_driver = DrivingAI()

# Define possible driving actions
actions = ["accelerate", "brake", "swerve_left", "swerve_right"]

# Simulated driving session (10 actions)
for i in range(10):
    current_state = "highway"  # Example: Driving on a highway
    action = ai_driver.choose_action(current_state, actions)

    # Simulate environment feedback (reward or penalty)
    if action == "accelerate":
        reward = 10 if random.random() > 0.2 else -5  # Safe 80% of the time
    elif action == "brake":
        reward = 8  # Generally safe action
    else:  # Swerving is risky
        reward = -10 if random.random() > 0.5 else 5

    next_state = "highway"  # Assume steady conditions
    ai_driver.update_q_table(current_state, action, reward, next_state)

    # Decide whether to reinforce or unlearn
    ai_driver.decide_reinforce_or_unlearn(action, reward)

print("Final Learned Q-table:", ai_driver.q_table)
