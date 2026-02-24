import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from twisted_bilayer_graphene import (
    reciprocal_vectors,
    build_sparse_H_tbg_k,
    theta_from_mn,
    generate_layer1_in_cell,
    wrap_to_cell,
    rot,
)

def dos_from_kmesh(pos1, subl1, pos2, subl2, t1, t2,
                   Nk=17, k_eigs=40, sigma_e=0.008,
                   Emin=-0.15, Emax=0.15, nE=1200, sigma_shift=1e-4):
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
    pref = 1.0 / (np.sqrt(2*np.pi) * sigma_e)

    for e0 in evals_all:
        dos += pref * np.exp(-0.5 * ((E - e0)/sigma_e)**2)

    dos /= (Nk**2)
    return E, dos, evals_all

def plot_dos(dos_angles=((5,6), (8,9), (10,11), (14,15))):
    plt.figure(figsize=(7, 4))
    for (m, n) in dos_angles:
        theta = theta_from_mn(m, n)
        theta_deg = np.degrees(theta)

        pos1, subl1, t1, t2 = generate_layer1_in_cell(m, n)
        pos2 = wrap_to_cell(pos1 @ rot(theta).T, t1, t2)
        subl2 = subl1.copy()

        E, dos, _ = dos_from_kmesh(
            pos1, subl1, pos2, subl2, t1, t2,
            Nk=17, k_eigs=40, sigma_e=0.008,
            Emin=-0.15, Emax=0.15, nE=1200
        )

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

def main():
    ap = argparse.ArgumentParser(description="TBG DOS plot (won't run on import).")

    ap.add_argument("--angles", nargs="+", default=["5,6", "8,9", "10,11", "14,15"],
                    help="List of m,n pairs like 5,6 8,9 ...")

    args = ap.parse_args()
    parsed = []
    for s in args.angles:
        m, n = s.split(",")
        parsed.append((int(m), int(n)))

    plot_dos(tuple(parsed))

if __name__ == "__main__":
    main()
