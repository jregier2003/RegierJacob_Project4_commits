import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    """
    Solve the 1D time-dependent Schrödinger equation.

    Parameters:
        nspace (int): Number of spatial grid points.
        ntime (int): Number of time steps to evolve.
        tau (float): Time step size.
        method (str): Evolution method ('ftcs' or 'crank'). Default 'ftcs'.
        length (float): Length of the spatial domain. Default 200.
        potential (list): Indices with non-zero potential. Default [].
        wparam (list): Parameters for initial Gaussian wave packet [sigma0, x0, k0]. Default [10, 0, 0.5].

    Returns:
        psi (2D array): Wave function values at each grid point and time step.
        x (1D array): Spatial grid points.
        t (1D array): Time grid points.
        prob (1D array): Total probability at each time step.
    """
    # Define spatial and time grids
    dx = length / nspace
    x = np.linspace(-length / 2, length / 2, nspace)
    t = np.arange(0, ntime * tau, tau)

    # Initialize the wave function with a Gaussian wave packet
    sigma0, x0, k0 = wparam
    psi = np.zeros((ntime, nspace), dtype=complex)
    psi[0, :] = np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)
    psi[0, :] /= np.sqrt(np.sum(np.abs(psi[0, :])**2) * dx)  

    # Initialize the potential array
    V = np.zeros(nspace)
    for idx in potential:
        if 0 <= idx < nspace:
            V[idx] = 1

    if method == 'ftcs':
        # FTCS method: Explicit finite-difference time evolution
        alpha = tau / dx**2
        #Ensure Stability
        alpha = adjust_alpha_for_stability(alpha, nspace)  

        # Time evolution loop
        for n in range(ntime - 1):
            for i in range(1, nspace - 1):
                psi[n + 1, i] = psi[n, i] - 1j * alpha * (psi[n, i + 1] - 2 * psi[n, i] + psi[n, i - 1]) + 1j * tau * V[i] * psi[n, i]
            # Apply periodic boundary conditions
            psi[n + 1, 0] = psi[n + 1, -2]
            psi[n + 1, -1] = psi[n + 1, 1]
    elif method == 'crank':
        # Crank-Nicholson method: Implicit time evolution
        alpha = 1j * tau / (2 * dx**2)
        beta = 1j * tau / 2
        # Construct tridiagonal matrices A and B
        A = create_tridiagonal_matrix(nspace, -alpha, 1 + 2 * alpha + beta * V, -alpha)
        B = create_tridiagonal_matrix(nspace, alpha, 1 - 2 * alpha - beta * V, alpha)

        # Time evolution loop
        for n in range(ntime - 1):
            rhs = B @ psi[n, :]
            psi[n + 1, :] = solve(A, rhs)

    # Compute total probability at each time step
    prob = np.zeros(ntime)
    for n in range(ntime):
        prob[n] = np.sum(np.abs(psi[n, :])**2) * dx
    return psi, x, t, prob

def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    """
    Create a tridiagonal matrix with specified diagonal and off-diagonal values.

    Parameters:
        size (int): Size of the square matrix.
        below_diag (complex): Value for elements below the diagonal.
        diag (complex): Value for diagonal elements.
        above_diag (complex): Value for elements above the diagonal.

    Returns:
        matrix (2D array): Constructed tridiagonal matrix.
    """
    matrix = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix

def compute_spectral_radius(matrix):
    """
    Compute the spectral radius (maximum absolute eigenvalue) of a matrix.

    Parameters:
        matrix (2D array): The matrix for which to compute the spectral radius.

    Returns:
        float: Spectral radius of the matrix.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

def adjust_alpha_for_stability(alpha, nspace):
    """
    Adjust the alpha parameter dynamically to ensure FTCS stability.

    Parameters:
        alpha (float): Initial alpha value (based on tau and dx).
        nspace (int): Number of spatial grid points.

    Returns:
        float: Adjusted alpha value that ensures stability.
    """
    max_attempts = 100
    adjustment_factor = 0.9 

    for _ in range(max_attempts):
        evolution_matrix = create_tridiagonal_matrix(nspace, -alpha, 1 + 2 * alpha, -alpha)
        radius = compute_spectral_radius(evolution_matrix)
        if radius <= 1:
            print(f"Adjusted alpha to: {alpha} (Spectral Radius: {radius})")
            return alpha
        alpha *= adjustment_factor

    print("Warning: Could not stabilize FTCS.")
    return alpha

def sch_plot(psi, x, t, plot_type='psi', time_index=0):
    """
    Plot the results of the function that solves Schrödinger equation.

    Parameters:
        psi (2D array): Wave function data.
        x (1D array): Spatial grid points.
        t (1D array): Time grid points.
        plot_type (str): Type of plot ('psi' for real part of wave function, 'prob' for probability density).
        time_index (int): Time index for the plot. Default 0.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    if plot_type == 'psi':
        plt.plot(x, np.real(psi[time_index, :]), label='Real(ψ)', lw=2)
        plt.title(f"Wave Function (Real Part) at t={t[time_index]:.2f}")
        plt.ylabel("Amplitude")
    elif plot_type == 'prob':
        plt.plot(x, np.abs(psi[time_index, :])**2, label='|ψ|²', lw=2)
        plt.title(f"Probability Density at t={t[time_index]:.2f}")
        plt.ylabel("Probability")
    plt.xlabel("x")
    plt.legend()
    plt.grid()
    plt.show()

def test_schrodinger():
    """
    Test the Schrödinger equation solver using FTCS and Crank-Nicholson methods.

    Parameters:
        None

    Returns:
        None
    """
    test_params = [
        {"tau": 0.0025, "nspace": 150},
        {"tau": 0.002, "nspace": 200},
        {"tau": 0.0015, "nspace": 250},
    ]

    # Test FTCS with varying parameters
    for params in test_params:
        print(f"\nTesting with tau={params['tau']}, nspace={params['nspace']}")
        try:
            psi_ftcs, x, t, prob_ftcs = sch_eqn(nspace=params['nspace'], ntime=500, tau=params['tau'], method='ftcs')
            print("FTCS Total Probability (Last Step):", prob_ftcs[-1])
            sch_plot(psi_ftcs, x, t, plot_type='prob', time_index=250)
        except ValueError as e:
            print("FTCS Test Failed:", e)

    # Test Crank-Nicholson with default parameters
    psi_crank, x, t, prob_crank = sch_eqn(nspace=100, ntime=500, tau=0.01, method='crank')
    print("\nCrank-Nicholson Total Probability (Last Step):", prob_crank[-1])
    sch_plot(psi_crank, x, t, plot_type='prob', time_index=250)

# Run the tests
test_schrodinger()

