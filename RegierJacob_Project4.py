import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    dx = length / nspace
    x = np.linspace(-length / 2, length / 2, nspace)
    t = np.arange(0, ntime * tau, tau)

    sigma0, x0, k0 = wparam
    psi = np.zeros((ntime, nspace), dtype=complex)
    psi[0, :] = np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)
    psi[0, :] /= np.sqrt(np.sum(np.abs(psi[0, :])**2) * dx) 
    return psi, x, t, None




