from qiskit.circuit.library import MCMT
from qiskit.circuit.library import GroverOperator

def grover_search(num_qubits=3):
    """ Implements Grover's Algorithm for 3-qubit database search. """
    qc = QuantumCircuit(num_qubits)

    # Step 1: Initialize state in equal superposition
    qc.h(range(num_qubits))

    # Step 2: Oracle (Marks |101⟩)
    oracle = QuantumCircuit(num_qubits)
    oracle.x([0, 2])
    oracle.append(MCMT('x', num_qubits - 1, 1), range(num_qubits))
    oracle.x([0, 2])

    # Step 3: Grover Diffusion Operator
    diffusion = QuantumCircuit(num_qubits)
    diffusion.h(range(num_qubits))
    diffusion.x(range(num_qubits))
    diffusion.append(MCMT('x', num_qubits - 1, 1), range(num_qubits))
    diffusion.x(range(num_qubits))
    diffusion.h(range(num_qubits))

    # Apply Grover iteration
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)

    qc.measure_all()
    return qc

# Simulate Grover's Search
grover_circuit = grover_search()
simulator = Aer.get_backend('qasm_simulator')
result = execute(grover_circuit, simulator, shots=1024).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
grover_circuit.draw("mpl")
