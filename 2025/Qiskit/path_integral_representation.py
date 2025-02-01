from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# Define a 4-qubit system (4 slits)
qc = QuantumCircuit(4, 4)

# Superposition of paths (All slits open)
qc.h(range(4))

# Path-dependent phase shifts (Feynman's action contributions)
qc.p(np.pi / 4, 0)  # Path 1
qc.p(np.pi / 3, 1)  # Path 2
qc.p(np.pi / 2, 2)  # Path 3
qc.p(np.pi, 3)      # Path 4

# Interference effect (Recombining paths)
qc.h(range(4))

# Measurement
qc.measure(range(4), range(4))

# Run simulation
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

# Plot histogram
plot_histogram(counts)
plt.title("Series of Double-Slit Experiments (Path Integral Simulation)")
plt.show()
