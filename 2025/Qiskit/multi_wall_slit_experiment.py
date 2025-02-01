from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


def multi_wall_slit_experiment(num_walls):
    """
    Simulates a quantum multi-wall double-slit experiment.
    
    Args:
        num_walls (int): Number of walls, each with 2 slits.
    
    Returns:
        Histogram of quantum measurement results.
    """
    num_qubits = num_walls  # Each qubit represents a choice at a specific wall
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Step 1: Initial superposition (particle can pass through both slits at each wall)
    qc.h(range(num_qubits))

    # Step 2: Path-dependent phase shifts at each wall
    for i in range(num_qubits):
        phase_shift = (i + 1) * np.pi / (num_walls + 1)  # Assign unique phase per wall
        qc.p(phase_shift, i)

    # Step 3: Interference (Final recombination)
    qc.h(range(num_qubits))

    # Step 4: Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    # Simulate experiment
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1000).result()
    counts = result.get_counts()
    
    # Plot histogram of results
    plot_histogram(counts)
    plt.title(f"Quantum Interference Through {num_walls} Walls")
    plt.show()
    
    return qc

# Running the experiment for increasing number of walls
for walls in range(2, 7):
    print(f"\n=== Multi-Wall Quantum Interference with {walls} Walls ===")
    circuit = multi_wall_slit_experiment(walls)
    print(circuit.draw())  # Display quantum circuit
