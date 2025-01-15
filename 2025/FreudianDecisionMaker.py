```python
import random

class FreudianDecisionMaker:
    def __init__(self, id_weight, ego_weight, superego_weight):
        self.id_weight = id_weight
        self.ego_weight = ego_weight
        self.superego_weight = superego_weight

    def make_decision(self, scenario):
        # Assign scores based on scenario
        id_score = random.uniform(0, 1) * self.id_weight
        ego_score = random.uniform(0, 1) * self.ego_weight
        superego_score = random.uniform(0, 1) * self.superego_weight

        scores = {
            "id": id_score,
            "ego": ego_score,
            "superego": superego_score
        }

        # Determine dominant force
        dominant = max(scores, key=scores.get)
        return f"In this scenario, the {dominant} dominates with a score of {scores[dominant]:.2f}."

# Define weights for id, ego, and superego
weights = {
    "hunger_dilemma": (0.35, 0.30, 0.35),
    "job_interview": (0.35, 0.35, 0.30),
    "moral_dilemma": (0.30, 0.35, 0.35)
}

# Instantiate decision-maker for each scenario
decision_makers = {k: FreudianDecisionMaker(*v) for k, v in weights.items()}

# Test scenarios
for scenario, maker in decision_makers.items():
    print(f"Scenario: {scenario.replace('_', ' ').title()}")
    print(maker.make_decision(scenario))
    print()
``
