from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.library import thermal_relaxation_error
from qiskit import QuantumCircuit

# Create a simple quantum circuit
qc = QuantumCircuit(1, 1)
qc.h(0)  # Start with a superposition state
qc.measure(0, 0)

# Define noise model: thermal relaxation (dissipation)
noise_model = AerSimulator().from_backend(Aer.get_backend("aer_simulator"))

# Simulate with noise
result_noisy = execute(qc, noise_model, shots=1000).result()
counts_noisy = result_noisy.get_counts()

# Plot results
plot_histogram(counts_noisy)
plt.title("Dissipative Open Quantum System (Thermal Relaxation)")
plt.show()
