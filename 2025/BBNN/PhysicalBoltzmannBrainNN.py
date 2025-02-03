import numpy as np

class PhysicalBoltzmannBrainNN:
    def __init__(self, input_size, hidden_size, output_size, forehead_temperature=310.15, natural_unit=1.0, boltzmann_constant=1.38e-23):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.temperature = forehead_temperature  # Use forehead temperature in Kelvin
        self.natural_unit = natural_unit  # Introduce natural unit for scaling
        self.boltzmann_constant = boltzmann_constant  # Introduce Boltzmann constant
        
        # Initialize weights randomly
        self.W_input_hidden = np.random.randn(input_size, hidden_size) * self.natural_unit
        self.W_hidden_output = np.random.randn(hidden_size, output_size) * self.natural_unit
        
    def boltzmann_distribution(self, x):
        exp_x = np.exp(-x / (self.temperature * self.natural_unit * self.boltzmann_constant))
        return exp_x / np.sum(exp_x)
    
    def sample_activation(self, x):
        prob = self.boltzmann_distribution(x)
        return (np.random.rand(*prob.shape) < prob).astype(float)
    
    def forward(self, x):
        hidden_input = np.dot(x, self.W_input_hidden)
        hidden_output = self.sample_activation(hidden_input)
        
        output_input = np.dot(hidden_output, self.W_hidden_output)
        output = self.sample_activation(output_input)
        
        return output
    
# Example usage
np.random.seed(42)  # For reproducibility
bbnn = PhysicalBoltzmannBrainNN(input_size=3, hidden_size=5, output_size=2, forehead_temperature=310.15, natural_unit=1.0, boltzmann_constant=1.38e-23)

# Test with a random input
input_data = np.array([1, 0, 1])
output = bbnn.forward(input_data)
print("Output:", output)
