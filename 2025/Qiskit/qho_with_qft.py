from qiskit.circuit.library import QFT

def qho_with_qft(n_qubits=4):
    """Simulates a quantum harmonic oscillator and applies a Quantum Fourier Transform."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Initialize in a superposition of energy eigenstates
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Step 2: Apply phase shifts to mimic time evolution
    for qubit in range(n_qubits):
        qc.p((qubit + 1) * np.pi / 4, qubit)

    # Step 3: Apply QFT to simulate wavefunction evolution in momentum space
    qc.append(QFT(n_qubits), range(n_qubits))

    # Step 4: Measure in QFT basis
    qc.measure(range(n_qubits), range(n_qubits))

    return qc

# Run the QHO with QFT
qho_qft_circuit = qho_with_qft(4)
job = execute(qho_qft_circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
qho_qft_circuit.draw("mpl")
