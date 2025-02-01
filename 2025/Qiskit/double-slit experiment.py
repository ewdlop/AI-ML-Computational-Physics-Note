from qiskit import QuantumCircuit

# Define a quantum circuit with two paths (qubits) representing the two slits
qc = QuantumCircuit(2, 2)

# Apply Hadamard to create superposition (particle taking both slits)
qc.h(0)

# CNOT to entangle path qubit with detection qubit
qc.cx(0, 1)

# Apply Hadamard again to create interference
qc.h(0)

# Measure the system
qc.measure([0, 1], [0, 1])

# Simulate
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts(qc)

# Plot results
plot_histogram(counts)
plt.title("Double-Slit Quantum Interference")
plt.show()
