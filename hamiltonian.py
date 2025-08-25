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

sigma_z_1 = pauli_z

sigma_z_2 = np.kron(pauli_0, pauli_z)

tau_1 = np.array([pauli_x, pauli_y])

tau_x_2 = np.kron(pauli_x, pauli_0)
tau_y_2 = np.kron(pauli_y, pauli_0)
tau_2 = np.array([tau_x_2, tau_y_2])

well_length = 500 # nm
well_width = 50 # nm
well_height = 10 # nm

a_x = 5 # nm
a_y = 5 # nm
s_x = a_x / np.sqrt(2)
s_y = a_y / np.sqrt(2)
s = (s_x, s_y, 0)

sigma_delta = 30

# ueV, ueV, (nm, nm) ?
Delta_R = np.random.normal(0.0, sigma_delta, (well_length, well_width))
Delta_J = np.random.normal(0.0, sigma_delta, (well_length, well_width))
Delta = np.moveaxis([Delta_R, Delta_J], 0, -1)
Delta = scipy.ndimage.gaussian_filter(Delta, sigma=s) 

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

f_B = 10 # MHz
timestep = 0.000002 # ?
h = 0 # Js / eVs ?
h_bar = h / (2 * np.pi)
J_to_eV = 0 # e = 1.602176634e−19

Delta_E_B = 0.041 # ueV ~= 10 MHz?

def hamiltonian(Delta, E_B):
    n_hat_Delta = Delta / np.linalg.norm(Delta)
    H = (E_B / 2) * sigma_z_2 + np.tensordot(Delta, tau_2, axes=1) + np.kron((Delta_E_B / 4) * np.tensordot(n_hat_Delta, tau_1, axes=1), sigma_z_1)
    return H

vec_hamiltonian = np.vectorize(hamiltonian) # ?

H = [[hamiltonian(Delta[x,y], E_B) for y in range(0, well_width)] for x in range(0, well_length)]

eigvals_H, eigvecs_H = np.linalg.eigh(H)
print(eigvals_H.shape)
plt.imshow(eigvals_H[:,:,0])
plt.show()