import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

a = 2.46
gamma0 = 3

# Reciprocal grid
K = (4*np.pi)/(3*a)
N = 1000

kx = np.linspace(-K, K, N)
ky = np.linspace(-K, K, N)
KX, KY = np.meshgrid(kx, ky)

# No twist angle

def H_mono(kx,ky):
    
    H = -gamma0 * np.array([[0, np.sqrt(3)*a/2*(kx - 1j*ky)],[np.sqrt(3)*a/2*(kx + 1j*ky), 0]])
    return H

# Add twist angle
def R(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def rotate_momentum_space(kx,ky,theta):
    return R(theta) @ np.array([kx, ky])

def H_bilayer_twist_no_coupling(kx,ky,theta):
    
    k1 = rotate_momentum_space(kx, ky, theta/2)  # Layer 1
    k2 = rotate_momentum_space(kx, ky, -theta/2) # Layer 2
    
    H1 = H_mono(k1[0], k1[1])
    H2 = H_mono(k2[0], k2[1])
    
    H = np.block([[H1, np.zeros_like(H1)], [np.zeros_like(H2), H2]])
    
    return H

# At each (kx,ky) point, as H is 4x4, you will get 4 eigenvalues. These are the 4 bands
# bands needs to be a 3D array of shape (N, N, 4) for N points in kx and ky.

bands = np.zeros((N, N, 4))
theta = 1.05 * np.pi / 180  

for i in range(N):
    for j in range(N):
        H = H_bilayer_twist_no_coupling(KX[i, j], KY[i, j], theta) # example twist angle
        eigenvalues = LA.eigh(H)[0]  # Only return eigenvalues, not eigenvectors
        bands[i, j, :] = eigenvalues
colourmap = ['viridis', 'inferno', 'plasma', 'cividis']  # Example colormaps for the 4 bands
# Plotting the bands
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for n in range(4):
    ax.plot_surface(KX, KY, bands[:, :, n], colourmap[n], alpha=0.8)
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-10, 10])
ax.set_xlabel('kx (1/Å)', fontsize=12)
ax.set_ylabel('ky (1/Å)', fontsize=12)
ax.set_zlabel('Energy (eV)', fontsize=12)
ax.view_init(elev=8, azim=10)
plt.show()


