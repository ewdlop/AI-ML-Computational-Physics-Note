from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def epr_bell_experiment():
    """ Simulates the EPR-Bell test experiment. """
    qc = QuantumCircuit(2, 2)

    # Step 1: Create an EPR (Bell) pair
    qc.h(0)  # Hadamard gate on qubit 0
    qc.cx(0, 1)  # Entanglement (CNOT)

    # Step 2: Alice and Bob measure in different bases
    qc.measure(0, 0)  # Alice measures in standard basis
    qc.measure(1, 1)  # Bob measures in standard basis

    return qc

# Simulate the EPR experiment
epr_circuit = epr_bell_experiment()
simulator = Aer.get_backend('qasm_simulator')
result = execute(epr_circuit, simulator, shots=1000).result()
counts = result.get_counts()

# Display results
plot_histogram(counts)
epr_circuit.draw("mpl")
