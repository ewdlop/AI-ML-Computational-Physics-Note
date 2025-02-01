# Quantum circuit to test the uncertainty principle
qc = QuantumCircuit(1, 2)

# Prepare state
qc.h(0)  # Put qubit in superposition (uncertainty between X and Z)

# Measure in X-basis (Hadamard + Z-measurement)
qc.h(0)
qc.measure(0, 0)

# Measure in Z-basis
qc.measure(0, 1)

# Run simulation
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)
plt.title("Uncertainty Principle Test (X and Z Measurement)")
plt.show()
