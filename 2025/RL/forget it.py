class ReverseRL:
    def __init__(self):
        self.bad_action_memory = {}
    
    def track_action(self, action, outcome):
        """Store how often an action leads to a negative outcome"""
        if outcome < 0:  # Negative reinforcement
            self.bad_action_memory[action] = self.bad_action_memory.get(action, 0) + 1

    def unlearn(self):
        """Reduce the probability of selecting bad actions"""
        for action, penalty in self.bad_action_memory.items():
            if penalty > 5:  # If action led to bad outcomes multiple times
                print(f"Unlearning bad action: {action}")
                self.bad_action_memory[action] = max(0, penalty - 1)  # Decay bad habit

# Example Usage
rrl = ReverseRL()

# AI takes actions (some good, some bad)
actions = ["speed_up", "slow_down", "swerve_left", "swerve_right"]
outcomes = [10, 5, -10, -15]  # Negative values are bad habits

# Track actions
for action, outcome in zip(actions, outcomes):
    rrl.track_action(action, outcome)

# Unlearn bad habits
rrl.unlearn()
