from qiskit import QuantumCircuit
import numpy as np

def bb84_protocol(num_bits=10):
    """ Implements the BB84 quantum key distribution protocol. """
    alice_bits = np.random.randint(2, size=num_bits)  # Alice's random bit string
    alice_bases = np.random.randint(2, size=num_bits)  # Alice's random bases
    bob_bases = np.random.randint(2, size=num_bits)  # Bob's random bases

    qc = QuantumCircuit(num_bits, num_bits)

    # Alice encodes qubits
    for i in range(num_bits):
        if alice_bits[i] == 1:
            qc.x(i)  # Encode |1>
        if alice_bases[i] == 1:
            qc.h(i)  # Encode in diagonal basis

    # Bob's measurement in random bases
    for i in range(num_bits):
        if bob_bases[i] == 1:
            qc.h(i)
        qc.measure(i, i)

    return qc, alice_bits, alice_bases, bob_bases

# Simulate BB84
bb84_circuit, alice_bits, alice_bases, bob_bases = bb84_protocol(10)
simulator = Aer.get_backend('qasm_simulator')
result = execute(bb84_circuit, simulator, shots=1).result()
bob_results = list(result.get_counts().keys())[0]

# Determine key based on matching bases
key = [int(bob_results[i]) for i in range(10) if alice_bases[i] == bob_bases[i]]

print("Final Secure Key:", key)
bb84_circuit.draw("mpl")
