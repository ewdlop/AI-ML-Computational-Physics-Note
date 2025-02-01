from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_state_city

# Define a quantum circuit for a pure state |ψ⟩ = (|0⟩ + |1⟩)/√2
qc_pure = QuantumCircuit(1)
qc_pure.h(0)  # Apply Hadamard to create superposition

# Simulate the pure state
simulator = Aer.get_backend('statevector_simulator')
job = execute(qc_pure, simulator)
result = job.result()
statevector = result.get_statevector()

# Convert to density matrix
rho_pure = DensityMatrix(statevector)

# Plot pure state density matrix
plot_state_city(rho_pure, title="Pure State Density Matrix")
plt.show()

# Create a mixed state: 50% |0⟩⟨0| + 50% |1⟩⟨1|
rho_mixed = 0.5 * DensityMatrix.from_label('0') + 0.5 * DensityMatrix.from_label('1')

# Plot mixed state density matrix
plot_state_city(rho_mixed, title="Mixed State Density Matrix")
plt.show()
