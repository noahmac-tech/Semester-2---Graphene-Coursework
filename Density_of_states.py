import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from twisted_bilayer_graphene import reciprocal_vectors
from twisted_bilayer_graphene import build_sparse_H_tbg_k
from twisted_bilayer_graphene import theta_from_mn, generate_layer1_in_cell, wrap_to_cell, rot



# Section D

def dos_from_kmesh(pos1, subl1, pos2, subl2, t1,t2, Nk=17, k_eigs = 40, sigma_e = 0.008, Emin = -0.15, Emax=0.15, nE=1200, sigma_shift=1e-4):

    b1, b2 = reciprocal_vectors(t1, t2)

    us = np.linspace(0, 1, Nk, endpoint=False)
    vs = np.linspace(0, 1, Nk, endpoint=False)

    evals_all = []

    for u in us:
        for v in vs:
            kvec = u*b1 + v*b2
            Hk = build_sparse_H_tbg_k(pos1, subl1, pos2, subl2, t1, t2, kvec)
            
            ev = eigsh(Hk, k=k_eigs, sigma=sigma_shift, which="LM",
                        return_eigenvectors=False, maxiter=5000)
            evals_all.append(np.real(ev))

    evals_all = np.concatenate(evals_all)

    E = np.linspace(Emin, Emax, nE)
    dos = np.zeros_like(E)
    pref = 1.0 / ((np.sqrt(2*np.pi) * sigma_e))  # normalization prefactor
    
    for e0 in evals_all:
        dos += pref * np.exp(-0.5 * ((E - e0)/sigma_e)**2)
    
    dos /= (Nk**2)  # average over k-points

    return E, dos, evals_all

def first_prominent_peak(E, dos, E_window=(0.02, 0.20), rel_height=0.25):
    mask = (E >= E_window[0]) & (E <= E_window[1])
    Em = E[mask]
    dm = dos[mask]

    peaks = np.where((dm[1:-1] > dm[:-2]) & (dm[1:-1] > dm[2:]))[0] + 1
    if peaks.size == 0:
        j = np.argmax(dm)
        return float(Em[j]), float(dm[j])

    # “prominent” means peak higher than (min + rel_height*(max-min)) in window
    dmin, dmax = dm.min(), dm.max()
    thresh = dmin + rel_height * (dmax - dmin)

    peaks = peaks[dm[peaks] >= thresh]
    if peaks.size == 0:
        # fallback: choose highest
        j = np.argmax(dm)
        return float(Em[j]), float(dm[j])

    # choose the one closest to zero among prominent peaks
    j = peaks[np.argmin(Em[peaks])]
    return float(Em[j]), float(dm[j])


dos_angles = [(5,6), (8,9), (10,11), (14,15)]   # adjust as needed

theta_list = []
Evhs_list = []
Dvhs_list = []

plt.figure(figsize=(7,4))

for (m,n) in dos_angles:
    theta = theta_from_mn(m,n)
    theta_deg = np.degrees(theta)

    pos1, subl1, t1, t2 = generate_layer1_in_cell(m,n)
    pos2 = wrap_to_cell(pos1 @ rot(theta).T, t1, t2)
    subl2 = subl1.copy()

    E, dos, _ = dos_from_kmesh(pos1,subl1,pos2,subl2,t1,t2,
                               Nk=17, k_eigs=40, sigma_e=0.008,
                               Emin=-0.15, Emax=0.15, nE=1200)

    # plot DOS curve
    plt.plot(E, dos, lw=1.5, label=f"{theta_deg:.2f}°")



plt.axvline(0, alpha=0.3)
plt.xlabel("Energy E (eV)")
plt.ylabel("DOS (arb. units)")
plt.title("DOS near charge neutrality vs twist angle")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("TBG_DOS_comparison.png", bbox_inches='tight', pad_inches=0.1)
plt.show()




