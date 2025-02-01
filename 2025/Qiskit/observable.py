from qiskit.quantum_info import Operator

# Hermitian Operator (Pauli-X)
H = Operator.from_label('X')
print("Hermitian Operator:", H.is_hermitian())

# Non-Hermitian Operator
NH = np.array([[1, 1j], [-1j, 1]])  # Non-Hermitian matrix
NH_op = Operator(NH)
print("Non-Hermitian Operator:", NH_op.is_hermitian())
