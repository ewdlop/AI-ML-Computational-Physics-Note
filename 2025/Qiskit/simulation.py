from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# Define a quantum circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Apply Hadamard gate to put qubit into superposition
qc.h(0)

# Apply a rotation to simulate time evolution
qc.rx(np.pi / 4, 0)

# Measure the qubit
qc.measure(0, 0)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Transpile and execute the circuit
compiled_circuit = transpile(qc, simulator)
job = execute(compiled_circuit, simulator, shots=1000)
result = job.result()

# Get the results and plot
counts = result.get_counts(qc)
plot_histogram(counts)
plt.show()

# Display the quantum circuit
print(qc.draw())
