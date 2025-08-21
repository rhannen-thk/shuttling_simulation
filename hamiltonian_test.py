import numpy as np
import scipy
import matplotlib.pyplot as plt

pauli_0 = np.array([
    [1, 0],
    [0, 1]
])

pauli_x = np.array([
    [0, 1],
    [1, 0]
])

pauli_y = np.array([
    [ 0 , -1j],
    [ 1j,  0 ]
])

pauli_z = np.array([
    [ 1,  0],
    [ 0, -1]
])

sigma_z = np.kron(pauli_0, pauli_z)

tau_x = np.kron(pauli_x, pauli_0)
tau_y = np.kron(pauli_y, pauli_0)
tau = np.array([tau_x, tau_y])

def hamiltonian(Delta, E_B):
    H = (E_B / 2) * sigma_z + np.tensordot(Delta, tau, axes=1)
    return H

H = [hamiltonian(np.array([0.0, 0.0]), x) for x in range(-4, 5, 1)]

eigvals_H, eigvecs_H = np.linalg.eigh(H)

plt.plot(eigvals_H)
plt.show()