from qiskit.quantum_info import Operator
import numpy as np

# Define a Non-Hermitian matrix
NH_matrix = np.array([[1, 1j], [-1j, 1]])

# Convert to Qiskit Operator
NH_op = Operator(NH_matrix)

# Check if the matrix is Hermitian
print("Is the operator Hermitian?", NH_op.is_hermitian())

# Compute Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(NH_matrix)
print("Eigenvalues of the Non-Hermitian Operator:", eigenvalues)