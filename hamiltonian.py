import numpy as np
import scipy
import scipy.ndimage
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

well_length = 500
well_width = 50
well_height = 10

# ueV, ueV, (nm, nm) ?
Delta_R = np.random.normal(0.0, 30, (well_length, well_width))
Delta_R = scipy.ndimage.gaussian_filter(Delta_R, sigma = 5)

Delta_J = np.random.normal(0.0, 30, (well_length, well_width))
Delta_J = scipy.ndimage.gaussian_filter(Delta_J, sigma = 5)

Delta = np.moveaxis([Delta_R, Delta_J], 0, -1)
#Delta = scipy.ndimage.gaussian_filter(Delta, sigma=5)
# replace with two point covariance rule?

plt.imshow(Delta[:, :, 0])
plt.show()
plt.imshow(Delta[:, :, 1])
plt.show()
plt.imshow(np.linalg.norm(Delta, axis=2))
plt.show()

g = 2 # approximate landé g factor
mu_B = 58 # approximate bohr magneton in ueV/T
B = 50 # approximate magnetic flux density in mT
E_B = g * mu_B * B
Delta_E_B = 0.041 # ueV ~= 10 MHz?

def hamiltonian(Delta, E_B):
    n_hat_Delta = Delta / np.linalg.norm(Delta)
    H = (E_B / 2) * sigma_z + np.tensordot(Delta, tau, axes=1)# + np.kron((Delta_E_B / 4) * np.tensordot(n_hat_Delta, tau, axes=1), sigma_z)
    return H

vec_hamiltonian = np.vectorize(hamiltonian)

H = [[hamiltonian(Delta[x,y], E_B) for y in range(0, well_width)] for x in range(0, well_length)]

eigvals_H, eigvecs_H = np.linalg.eigh(H)
print(eigvals_H.shape)
plt.imshow(eigvals_H[:,:,0])
plt.show()