from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

def multi_wall_decoherence_experiment(num_walls, decoherence_prob=0.1):
    """
    Simulates a quantum multi-wall double-slit experiment with decoherence.
    
    Args:
        num_walls (int): Number of walls, each with 2 slits.
        decoherence_prob (float): Probability of decoherence-induced measurement.
    
    Returns:
        Histogram of quantum measurement results.
    """
    num_qubits = num_walls  # Each qubit represents a choice at a specific wall
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: Initial superposition (Quantum state explores all paths)
    qc.h(range(num_qubits))

    # Step 2: Apply phase shifts (Path Integral Contribution)
    for i in range(num_qubits):
        phase_shift = (i + 1) * np.pi / (num_walls + 1)  # Assign unique phase per wall
        qc.p(phase_shift, i)

    # Step 3: Simulating Decoherence
    for i in range(num_qubits):
        if np.random.rand() < decoherence_prob:
            qc.measure(i, i)  # Early measurement collapses the state
            qc.reset(i)        # Reset qubit (environmental interaction)
            qc.h(i)            # Reintroduce superposition, but with phase loss

    # Step 4: Interference (Final recombination)
    qc.h(range(num_qubits))

    # Step 5: Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    # Simulate experiment
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1000).result()
    counts = result.get_counts()

    # Plot histogram of results
    plot_histogram(counts)
    plt.title(f"Quantum Decoherence Through {num_walls} Walls (p = {decoherence_prob})")
    plt.show()
    
    return qc

decoherence_levels = [0.0, 0.2, 0.5, 0.8]  # Different decoherence intensities

for decoherence in decoherence_levels:
    print(f"\n=== Multi-Wall Quantum Interference with Decoherence (p={decoherence}) ===")
    circuit = multi_wall_decoherence_experiment(num_walls=4, decoherence_prob=decoherence)
    print(circuit.draw())  # Display quantum circuit
