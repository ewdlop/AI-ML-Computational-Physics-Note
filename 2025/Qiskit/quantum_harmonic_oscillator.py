from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
import numpy as np

def quantum_harmonic_oscillator(n_qubits=4):
    """Expands the quantum harmonic oscillator simulation to higher energy levels."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Initialize superposition of energy states
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Step 2: Apply phase rotations (simulating time evolution)
    for qubit in range(n_qubits):
        qc.p((qubit + 1) * np.pi / 6, qubit)  # Adjusted phase shift for better accuracy

    # Step 3: Introduce ladder operators (a and a† approximations)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.h(0)

    # Step 4: Measure in computational basis
    qc.measure(range(n_qubits), range(n_qubits))

    return qc

# Run the extended simulation
qho_circuit = quantum_harmonic_oscillator(4)
simulator = Aer.get_backend('qasm_simulator')
job = execute(qho_circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
qho_circuit.draw("mpl")
