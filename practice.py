import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh

a = 2.46
gamma0 = 3


"""Define grid in reciprocal space"""
K = (4*np.pi)/(3*a)
N = 400

kx = np.linspace(-K,K,N)
ky = np.linspace(-K,K,N)

KX, KY = np.meshgrid(kx, ky)

"""Define Nearest Neighbour sites"""
"""Choosing 0 as origin"""

sites = np.array([
    [0, a/(np.sqrt(3))], # Up
    [a/2, -a/(2*np.sqrt(3))], #Down Right
    [-a/2, -a/(2*np.sqrt(3))]]) #Down Left

def f(kx, ky, sites):
    sum = np.zeros_like(KX, dtype=complex)
    for site in sites:
        rx, ry = site
        sum += np.exp(1j * (kx * rx + ky * ry))
    return sum

"""Calculate positive and negative surfaces"""
"""Plot on same 3D axes"""
