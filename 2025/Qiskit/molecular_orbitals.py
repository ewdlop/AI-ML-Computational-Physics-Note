from qiskit_nature.second_q.drivers import Molecule
from qiskit_nature.second_q.visualization import draw_molecule

# Define the H₂ molecule
molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]])

# Draw the molecular orbitals
draw_molecule(molecule)
