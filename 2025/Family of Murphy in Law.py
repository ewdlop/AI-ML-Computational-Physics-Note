from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# Define a quantum circuit with 3 qubits (representing Murphy's states)
qc = QuantumCircuit(3, 3)

# Step 1: Create Superposition of Success & Failure
qc.h(range(3))  # Hadamard gate puts qubits in superposition (both success and failure exist)

# Step 2: Entangle Murphy’s Failures (If one fails, they all fail)
qc.cx(0, 1)  # Entangle qubit 0 with qubit 1
qc.cx(1, 2)  # Entangle qubit 1 with qubit 2

# Step 3: Introduce Murphy’s Noise (Quantum "Bad Luck")
random_phase = np.random.uniform(0, 2 * np.pi, 3)
for i in range(3):
    qc.p(random_phase[i], i)  # Apply random phase noise to simulate uncertainty

# Step 4: Measurement (Collapse the Quantum Murphy's Law)
qc.measure(range(3), range(3))

# Simulate Murphy’s Quantum Failure
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
counts = result.get_counts()

# Output the "Quantum Murphy’s Law" Result
print("Quantum Murphy’s Law Outcome:", counts)

# Draw the quantum circuit
qc.draw('mpl')
