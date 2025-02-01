def photon_qho(n_qubits=3):
    """Simulates a quantum harmonic oscillator in a coherent state."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Create an initial superposition
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Step 2: Apply photon creation operator (a†)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Step 3: Apply displacement operator (simulating photon excitation)
    qc.rx(np.pi / 3, 0)

    # Step 4: Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc

# Run photon-based QHO simulation
photon_qho_circuit = photon_qho(3)
job = execute(photon_qho_circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
photon_qho_circuit.draw("mpl")
