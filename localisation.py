import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from twisted_bilayer_graphene import reciprocal_vectors
from twisted_bilayer_graphene import build_sparse_H_tbg_k
from twisted_bilayer_graphene import theta_from_mn, generate_layer1_in_cell, wrap_to_cell, rot





def low_energy_density_at_k(pos1, subl1, pos2, subl2, t1, t2, kvec,
                            n_states=4, k_eigs=40, sigma_shift=1e-4):

    Hk = build_sparse_H_tbg_k(pos1, subl1, pos2, subl2, t1, t2, kvec)

    evals, evecs = eigsh(Hk, k=k_eigs, sigma=sigma_shift,
                         which="LM", return_eigenvectors=True)

    evals = np.real(evals)
    idx = np.argsort(np.abs(evals))[:n_states]

    psi_block = evecs[:, idx]

    # summed density
    rho = np.sum(np.abs(psi_block)**2, axis=1)

    rho /= np.sum(rho)  # normalise

    return rho

def low_energy_density_avg_near_Gamma(pos1, subl1, pos2, subl2, t1, t2,
                                      n_states=4, k_eigs=40, sigma_shift=1e-4,
                                      k_frac=0.10):
    """
    Average summed low-energy density over a few k-points near Γ.
    k_frac sets how far from Γ in units of reciprocal vectors.
    """
    b1, b2 = reciprocal_vectors(t1, t2)

    k_list = [
        np.array([0.0, 0.0]),
        k_frac * b1,
        k_frac * b2,
        k_frac * (b1 + b2),
    ]

    rho_accum = None
    for kvec in k_list:
        rho = low_energy_density_at_k(pos1, subl1, pos2, subl2, t1, t2, kvec,
                                      n_states=n_states, k_eigs=k_eigs, sigma_shift=sigma_shift)
        if rho_accum is None:
            rho_accum = rho.copy()
        else:
            rho_accum += rho

    rho_accum /= len(k_list)
    rho_accum /= np.sum(rho_accum)  # renormalize
    return rho_accum

def ipr(psi):
    p = np.abs(psi)**2
    return float(np.sum(p**2))

def plot_density_tiled(pos1, pos2, rho, t1, t2, title="", vmin=None, vmax=None):
    N1 = len(pos1)
    rho1 = rho[:N1]
    rho2 = rho[N1:]
    rho_sum = rho1 + rho2

    shifts = [
        np.array([0,0]),
        t1,
        t2,
        t1 + t2
    ]

    fig, ax = plt.subplots(figsize=(6,5))

    for S in shifts:
        pos_shift = pos1 + S
        sc = ax.scatter(pos_shift[:,0], pos_shift[:,1],
                        c=rho_sum, s=10, marker="o",
                        vmin=vmin, vmax=vmax)

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(title)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\rho_1+\rho_2$")

    plt.tight_layout()
    plt.show()

angles_E = [(5,6), (14,15)]  # large angle vs small angle
states = []

for (m, n) in angles_E:
    theta = theta_from_mn(m, n)
    theta_deg = np.degrees(theta)

    pos1, subl1, t1, t2 = generate_layer1_in_cell(m, n)
    pos2 = wrap_to_cell(pos1 @ rot(theta).T, t1, t2)
    subl2 = subl1.copy()

    rho = low_energy_density_avg_near_Gamma(pos1, subl1, pos2, subl2, t1, t2,
                                        n_states=4, k_eigs=40, sigma_shift=1e-4,
                                        k_frac=0.10)

    states.append((theta_deg, pos1, pos2, rho, t1, t2))

# Use common color scale so localisation can be compared fairly
vmax = max(np.max(s[3][:len(s[1])] + s[3][len(s[1]):]) for s in states)

for theta_deg, pos1, pos2, rho, t1, t2 in states:
    plot_density_tiled(
        pos1, pos2, rho, t1, t2,
        title=rf"Summed low-energy density near $\Gamma$ (θ={theta_deg:.2f}°)",
        vmin=0.0, vmax=vmax
    )