from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
import numpy as np

def shor_code():
    """ Implements the 9-qubit Shor Code with both bit-flip and phase-flip error correction. """
    qc = QuantumCircuit(9, 9)  

    # Step 1: Encoding (Logical |0> → |000000000>, Logical |1> → |111111111>)
    qc.h(0)
    for i in range(1, 9, 3):  # Encoding first group of 3
        qc.cx(0, i)
    for i in range(3, 9, 3):  # Encoding second group of 3
        qc.cx(1, i)
    for i in range(6, 9):  # Encoding third group of 3
        qc.cx(2, i)

    # Step 2: Simulated Errors (Bit-flip and Phase-flip)
    qc.x(4)  # Bit-flip on qubit 4
    qc.z(6)  # Phase-flip on qubit 6

    # Step 3: Error Correction
    # Majority vote for bit-flip
    qc.cx(3, 7)
    qc.cx(3, 8)
    qc.measure(7, 0)
    qc.measure(8, 1)
    
    # Correction conditioned on majority vote
    qc.x(3).c_if(0, 1)
    qc.x(3).c_if(1, 1)

    # Majority vote for phase-flip
    qc.h(range(9))  # Change to Hadamard basis
    qc.cx(3, 7)
    qc.cx(3, 8)
    qc.measure(7, 2)
    qc.measure(8, 3)

    # Phase-flip correction
    qc.z(3).c_if(2, 1)
    qc.z(3).c_if(3, 1)

    return qc

# Simulate and visualize
qc_shor = shor_code()
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc_shor, simulator, shots=1000).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
qc_shor.draw("mpl")
