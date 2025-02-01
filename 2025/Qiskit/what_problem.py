from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Quantum Circuit with 2 Qubits
qc = QuantumCircuit(2, 2)

# Put first qubit in superposition (Unmeasured quantum state)
qc.h(0)

# Scenario 1: Collapse upon Z-measurement
qc.measure(0, 0)

# Scenario 2: Measure in X-basis (Superposition test)
qc.h(1)  # Change basis
qc.measure(1, 1)

# Run simulation
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
