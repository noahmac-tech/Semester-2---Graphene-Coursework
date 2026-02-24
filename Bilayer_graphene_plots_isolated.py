import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.constants import hbar, eV

a = 2.46e-10
gamma1 = 0.39 * eV
vF = 1e6  # m/s

def H_bilayer(qx, qy):
    return np.array([
        [0, vF * hbar * (qx + 1j * qy), 0, 0],
        [vF * hbar * (qx - 1j * qy), 0, gamma1, 0],
        [0, gamma1, 0, vF * hbar * (qx - 1j * qy)],
        [0, 0, vF * hbar * (qx + 1j * qy), 0],
    ])

def compute_bands(qmax=5e8, N=200):
    qx = np.linspace(-qmax, qmax, N)
    qy = np.linspace(-qmax, qmax, N)
    QX, QY = np.meshgrid(qx, qy)

    QX_nm = QX * 1e-9
    QY_nm = QY * 1e-9

    bands = np.zeros((4, N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            eigs_J = LA.eigvalsh(H_bilayer(QX[i, j], QY[i, j]))
            bands[:, i, j] = eigs_J / eV  # eV
    return QX_nm, QY_nm, bands

def plot_linecut(QX_nm, bands):
    N = QX_nm.shape[0]
    mid = N // 2  # qy=0 index

    plt.figure(figsize=(7, 4))
    for n in range(4):
        plt.plot(QX_nm[mid, :], bands[n, mid, :], label=f'Band {n+1}')
    plt.xlabel(r'$q_x$ (nm$^{-1}$) at $q_y=0$', fontsize=14)
    plt.ylabel('Energy (eV)', fontsize=14)
    plt.ylim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('Bilayer_Graphene_Bands_qy0.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_contour(QX_nm, QY_nm, bands):
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(QX_nm, QY_nm, bands[2], levels=40)  # low-energy conduction band
    plt.xlabel(r'$q_x$ (nm$^{-1}$)')
    plt.ylabel(r'$q_y$ (nm$^{-1}$)')
    plt.colorbar(cs, label='Energy (eV)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('Bilayer_Graphene_Contour.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_surfaces(QX_nm, QY_nm, bands):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(QX_nm, QY_nm, bands[0], cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
    ax.plot_surface(QX_nm, QY_nm, bands[1], cmap='inferno', alpha=0.5, linewidth=0, antialiased=True)
    ax.plot_surface(QX_nm, QY_nm, bands[2], cmap='plasma',  alpha=0.5, linewidth=0, antialiased=True)
    ax.plot_surface(QX_nm, QY_nm, bands[3], cmap='cividis', alpha=0.5, linewidth=0, antialiased=True)

    ax.set_xlabel(r'$q_x$ (nm$^{-1}$)')
    ax.set_ylabel(r'$q_y$ (nm$^{-1}$)')
    ax.set_zlabel('Energy (eV)', fontsize=12)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    ax.view_init(elev=8, azim=10)
    ax.set_box_aspect([1, 1, 1])
    ax.margins(0.1)

    plt.tight_layout()
    plt.savefig('Bilayer_Graphene_Band_Structure.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

QX_nm, QY_nm, bands = compute_bands()

plot_linecut(QX_nm, bands)
