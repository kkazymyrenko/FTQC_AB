import math
import numpy as np
from typing import Sequence
"""
Created on Mon Dec 12 10:10:37 2025

@author: kkyrylo
"""

def NegSVD(vec: Sequence[float], info: bool = False) -> float:
    """
    Compute the bipartite negativity between the first half and the last half
    of an N-qubit pure state, given by its amplitude vector `vec`.

    Assumptions:
    - len(vec) = 2^N for some integer N (i.e., vector length is a power of 2).
    - N is even (so the bipartition is into equal halves: N/2 qubits vs N/2 qubits).
    - vec is real (dtype=float); complex states would require handling complex arrays.

    Parameters
    ----------
    vec : Sequence[float]
        The state vector amplitudes (length must be a power of 2).
    info : bool, optional
        If True, prints the maximum possible negativity for this bipartition.

    Returns
    -------
    float
        The negativity value between the two halves (first N/2 qubits vs last N/2 qubits).
    """
    # Convert input to a NumPy float array
    v = np.array(vec, dtype=float)

    # Total number of amplitudes (should be 2^N)
    dim = v.size

    # Compute N from dimension using log2
    NQPU_float = math.log2(dim)

    # Check that N is an integer (i.e., dim is a power of 2)
    # Allow tiny floating errors in log2 by using isclose to nearest integer
    if not math.isclose(NQPU_float, round(NQPU_float), abs_tol=1e-9):
        raise ValueError("Length of vec must be a power of 2 (i.e., 2^N).")

    NQPU = int(round(NQPU_float))

    # Check that N is even so we can split into equal halves
    if (NQPU % 2) != 0:
        raise ValueError("Number of qubits N must be even to split into equal halves.")

    # Subsystem dimension d = 2^(N/2)
    d = 2 ** (NQPU // 2)

    # Reshape the state vector into a d x d matrix
    # This treats the amplitudes as a matrix whose SVD yields Schmidt coefficients (up to normalization).
    M = v.reshape(d, d)

    # Compute singular values only (faster than full U, Vt) — SVD of M
    S = np.linalg.svd(M, compute_uv=False)  # S is length d, nonnegative

    # Normalize the singular values according to the state vector norm.
    # If the state is normalized (||vec||_2 == 1), this does nothing.
    # If not, this scales S so that the singular values correspond to Schmidt coefficients of a normalized state.
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        raise ValueError("Input vector has zero norm; cannot compute negativity.")
    S = S / v_norm

    # Negativity for a pure bipartite state (equal bipartition):
    # N = ( (sum_i S_i)^2 - 1 ) / 2
    # Here S_i are the singular values (Schmidt coefficients) of the reshaped matrix M.
    negativity = 0.5 * ((np.sum(S) ** 2) - 1.0)

    # Maximum negativity for this bipartition is (d - 1)/2 (achieved by maximally entangled state)
    maxneg = 0.5 * (d - 1)

    if info:
        print(f"maximum negativity for {NQPU}-qubit system (split {NQPU//2}|{NQPU//2}) is {maxneg}")

    return float(negativity)


# 2-qubit maximally entangled (Bell) state: (|00> + |11>)/sqrt(2)
bell = np.array([1, 0, 0, 1], dtype=float)
bell /= np.sqrt(2)
print(NegSVD(bell, info=True))  # Should be 0.5 for d=2 (max negativity (2-1)/2 = 0.5)

# 4-qubit product of two Bell states -> maximally entangled across split 2|2
bell2 = np.kron(bell, bell)  # length 16
print(NegSVD(bell2, info=True))  # d = 2^(4/2) = 4 => max negativity = (4-1)/2 = 1.5

# Some examples of 0.5 negativities heat-maps

# 4-qubit boundary heat-map
AA = np.array([[1., 1., 1., 1.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [1., 1.,-1.,-1.]])
print(NegSVD(AA, info=True))  # d = 2^(4/2) = 4 => max negativity = (4-1)/2 = 1.5

# 4-qubit heat-map
AA = np.array([[1., 1., 1., 1.],
              [1., 1., 1., 1.],
              [1., 1.,-1.,-1.],
              [1., 1.,-1.,-1.]])
print(NegSVD(AA, info=True))  # d = 2^(4/2) = 4 => max negativity = (4-1)/2 = 1.5

# 12-qubit boundary heat-map with noise
Nsize = 2**6
AA = np.zeros((Nsize,Nsize))
AA[0] = [1]*(Nsize//2)+[0]*(Nsize//2)
AA[Nsize-1] = [0]*(Nsize//2)+[1]*(Nsize//2)
Arand = np.random.uniform(low=-1.e-1, high=1.e-1, size=(Nsize, Nsize))
AA += Arand
print(NegSVD(AA, info=True))
