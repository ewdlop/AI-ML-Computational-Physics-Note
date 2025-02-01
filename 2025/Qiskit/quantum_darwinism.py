from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def quantum_darwinism():
    """ Simulates Quantum Darwinism with system-environment entanglement. """
    qc = QuantumCircuit(3, 3)

    # Step 1: Put system qubit into superposition
    qc.h(0)

    # Step 2: Entangle system with environment qubits
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Step 3: Measure environment (simulating classical redundancy)
    qc.measure([1, 2], [1, 2])

    return qc

# Simulate Quantum Darwinism
qd_circuit = quantum_darwinism()
result = execute(qd_circuit, simulator, shots=1000).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
qd_circuit.draw("mpl")
