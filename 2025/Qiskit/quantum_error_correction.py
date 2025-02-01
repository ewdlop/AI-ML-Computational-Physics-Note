from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def encode_3_qubit_logical_state(qc, logical_qubit):
    """
    Encodes a single logical qubit using a 3-qubit repetition code.
    """
    qc.cx(logical_qubit, logical_qubit + 1)
    qc.cx(logical_qubit, logical_qubit + 2)

def introduce_bit_flip_noise(qc, error_probability=0.1):
    """
    Simulates bit-flip errors in the quantum circuit.
    """
    for qubit in range(qc.num_qubits):
        if np.random.rand() < error_probability:
            qc.x(qubit)  # Flip the qubit state (X error)

def error_correction(qc):
    """
    Detects and corrects bit-flip errors using majority voting.
    """
    # Measure the auxiliary qubits
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.measure(3, 0)  # Syndrome measurement

    # Correct errors using classical feedback
    qc.x(0).c_if(0, 1)  # If error detected, flip back
    qc.x(1).c_if(0, 1)
    qc.x(2).c_if(0, 1)

def quantum_error_correction(error_probability=0.1):
    """
    Full quantum error correction circuit with encoding, noise, and correction.
    """
    qc = QuantumCircuit(4, 1)  # 3 data qubits + 1 syndrome qubit
    
    # Step 1: Encode logical qubit
    qc.h(0)  # Initial superposition state
    encode_3_qubit_logical_state(qc, 0)
    
    # Step 2: Introduce noise (bit flips)
    introduce_bit_flip_noise(qc, error_probability)

    # Step 3: Apply error correction
    error_correction(qc)

    # Step 4: Measure final logical qubit
    qc.measure(0, 0)

    return qc

error_levels = [0.0, 0.1, 0.3, 0.5]  # Different noise probabilities

for error_prob in error_levels:
    print(f"\n=== Quantum Error Correction with Bit-Flip Probability {error_prob} ===")
    
    # Build and simulate circuit
    qec_circuit = quantum_error_correction(error_probability=error_prob)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qec_circuit, simulator, shots=1000).result()
    counts = result.get_counts()
    
    # Plot histogram
    plot_histogram(counts)
    plt.title(f"QEC Performance at p = {error_prob}")
    plt.show()
    
    print(qec_circuit.draw())


