from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# Load IBMQ account (Replace with your IBM Quantum API key)
IBMQ.save_account('YOUR_IBM_QUANTUM_API_KEY', overwrite=True)
IBMQ.load_account()

# Get least busy quantum backend with 9+ qubits
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 9 and not x.configuration().simulator))

print("Selected Backend:", backend.name())

# Transpile and execute the circuit on IBMQ
job = execute(shor_qec, backend, shots=1024)
job_monitor(job)
result_ibm = job.result()

# Retrieve and plot results
counts_ibm = result_ibm.get_counts()
plot_histogram(counts_ibm)
