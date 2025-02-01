from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt

def stern_gerlach_experiment(temp=0.1, axis='z'):
    """ Simulates the Stern-Gerlach experiment with temperature effects. """
    qc = QuantumCircuit(1, 1)

    # Step 1: Prepare superposition
    qc.h(0)  

    # Step 2: Introduce thermal randomness
    if np.random.rand() < temp:
        qc.x(0)  # Thermal flipping

    # Step 3: Measure spin in chosen axis
    if axis == 'x':
        qc.h(0)  # Rotate to X-basis
    elif axis == 'y':
        qc.sdg(0)
        qc.h(0)  # Rotate to Y-basis
    
    qc.measure(0, 0)

    return qc

# Simulate Stern-Gerlach
sg_circuit = stern_gerlach_experiment(temp=0.2, axis='x')
simulator = Aer.get_backend('qasm_simulator')
result = execute(sg_circuit, simulator, shots=1000).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
sg_circuit.draw("mpl")
