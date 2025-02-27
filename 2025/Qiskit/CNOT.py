from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram

# Create a Quantum Circuit with 2 qubits (representing gravity and quantum subsystems)
qc = QuantumCircuit(2)

# Step 1: Prepare an equal superposition for the 'Quantum Physics' qubit
qc.h(0)  # Apply Hadamard gate to create superposition

# Step 2: Entangle it with the 'Gravity' qubit
qc.cx(0, 1)  # Apply CNOT gate to entangle qubit 0 (quantum) with qubit 1 (gravity)

# Step 3: Measure both qubits
qc.measure_all()

# Visualize the circuit
print(qc.draw())

# Simulate the circuit
simulator = Aer.get_backend('aer_simulator')
tqc = transpile(qc, simulator)
qobj = assemble(tqc)
result = simulator.run(qobj).result()
counts = result.get_counts()

# Plot the results
plot_histogram(counts)
