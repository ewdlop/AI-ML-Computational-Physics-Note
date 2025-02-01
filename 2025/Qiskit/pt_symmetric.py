from qiskit.quantum_info import Operator

# Define a PT-symmetric matrix
H_PT = np.array([[0, 1j], [-1j, 0]])

# Convert to Qiskit Operator
H_PT_op = Operator(H_PT)

# Check if Hermitian
print("Is the PT-symmetric operator Hermitian?", H_PT_op.is_hermitian())

# Compute Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(H_PT)
print("Eigenvalues of PT-Symmetric Operator:", eigenvalues)
