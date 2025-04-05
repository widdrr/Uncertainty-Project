import numpy as np
from numpy.typing import NDArray
from functools import partial
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar

def entropy(x: np.float64) -> np.float64:
    """
    Compute the entropy of a given value.

    Args:
        x (np.float64): The input value.

    Returns:
        np.float64: The computed entropy.
    """

    ## Handle edge cases for pure and maximally mixed states.
    if x == 0 or x == 1:
        return np.float64(0.0)
    
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def purity_from_entropy(state_entropy: np.float64) -> Tuple[np.float64, np.float64]:
    """
    Numerically Compute the purity of a qubit given its entropy.

    For a qubit with eigenvalues p and 1-p, the von Neumann entropy is:
        S = -p*log2(p) - (1-p)*log2(1-p),
    and the purity is:
        P = p^2 + (1-p)^2.

    Args:
        state_entropy (np.float64): The entropy of the state.

    Returns:
        np.float64: The computed purity.
    """
    # Handle edge cases for pure and maximally mixed states.
    if state_entropy <= 0:
        return (np.float64(1.0), np.float64(1.0))
    if state_entropy >= 1:
        return (np.float64(0.5), np.float64(0.5))

    # Function to find the root: binary_entropy(λ) - state_entropy.
    def f(p: np.float64) -> np.float64:
        return entropy(p) - state_entropy

    # Use scipy's root_scalar with the bisection method in the range [0, 0.5].
    sol = root_scalar(f, bracket=(0.0, 0.5), method='bisect')
    p = sol.root

    purity_value = p**2 + (1 - p)**2
    return (np.float64(purity_value), np.float64(p))



def compute_boud(formula: np.ufunc, *args: NDArray[np.float64]) -> NDArray[np.float64]: # type: ignore
    """
    Compute the lower bound of uncertainty using a vectorized formula, applied elementwise on the vectorized inputs.

    Args:
        formula (np.ufunc): The vectorized formula to use for the lower bound.
        *args (NDArray[np.float64]): Vectors with the arguments for formula.

    Returns:
        NDArray[np.float64]: An array of computed lower bounds.
    """

    return np.clip(formula(*args), 0, None) # type: ignore[no-untyped-call]

def formula_4_func(entropy: np.float64, max_overlap: np.float64) -> np.float64:
    """
    Formula 4 for computing uncertainty lower bound.

    Args:
        entropy (np.float64): The entropy of the state.
        max_overlap (np.float64): The maximum overlap between measurement bases.

    Returns:
        np.float64: The computed lower bound.
    """
    return (entropy - 1) * np.log2(max_overlap)

def formula_12_func(state_entropy: np.float64, max_overlap: np.float64) -> np.float64:
    """
    Formula 12 for computing uncertainty lower bound.

    Args:
        state_entropy (np.float64): The entropy of the state.
        max_overlap (np.float64): The maximum overlap between measurement bases.

    Returns:
        np.float64: The computed lower bound.
    """
    return -np.log2(max_overlap) - state_entropy

def formula_13_func(state_entropy: np.float64, max_overlap: np.float64) -> np.float64:
    """
    Formula 13 for computing uncertainty lower bound.

    Args:
        state_entropy (np.float64): The entropy of the state.
        max_overlap (np.float64): The maximum overlap between measurement bases.

    Returns:
        np.float64: The computed lower bound.
    """
    return entropy((1 + np.sqrt(2* max_overlap - 1)) * 0.5) - 2 * state_entropy

def formula_14_func(state_entropy: np.float64, state_purity: np.float64, max_overlap: np.float64) -> np.float64:
    """
    Formula 14 for computing uncertainty lower bound.

    Args:
        state_entropy (np.float64): The entropy of the state.
        state_purity (np.float64): The purity of the state.
        max_overlap (np.float64): The maximum overlap between measurement bases.

    Returns:
        np.float64: The computed lower bound.
    """
    return entropy((np.sqrt(2 * state_purity - 1) * (2 * np.sqrt(max_overlap) - 1) + 1) * 0.5) - state_entropy

def optimal_func(coherence_func: Callable[[np.float64, np.float64],np.float64], eigenvalue: np.float64, max_overlap: np.float64) -> np.float64:
    """
    Optimal function for computing uncertainty lower bound.

    Args:
    coherence_func (Callable[[np.float64, np.float64],np.float64]): The function corresponding to the coherence measure used.
        state_entropy (np.float64): The entropy of the state.
        max_overlap (np.float64): The maximum overlap between measurement bases.

    Returns:
        np.float64: The computed lower bound.
    """

    gamma: np.float64 = np.arccos(2 * max_overlap - 1)

    def obj_func(alpha: np.float64) -> np.float64:
        return coherence_func(eigenvalue, (np.cos(alpha) + 1) * 0.5) + coherence_func(eigenvalue, (np.cos(gamma - alpha) + 1) * 0.5)

    res = minimize_scalar(obj_func, bounds=(gamma * 0.5, gamma), method='bounded') # type: ignore
    return res.fun # type: ignore

def relative_entropy_measure_func(eigenvalue: np.float64, x: np.float64) -> np.float64:
    """
    Function corresponding to relative entropy measure for computing the optimal uncertainty lower bound.

    Args:
        eigenvalue (np.float64): The eigenvalue of the state.
        x (np.float64): The input value.

    Returns:
        np.float64: The computed lower bound.
    """
    return entropy(eigenvalue * x + (1 - eigenvalue) * (1 - x)) - entropy(eigenvalue)


max_overlap_values = np.linspace(0.5, 1, 100).reshape(-1,1)
entropy_values = np.array([0, 0.3, 0.6, 0.9], dtype=np.float64).reshape(1,-1)

values = [purity_from_entropy(entropy) for entropy in entropy_values[0]]
purity_values = np.array([value[0] for value in values], dtype=np.float64).reshape(1,-1)
eigenvalue_values = np.array([value[1] for value in values], dtype=np.float64).reshape(1,-1)


formula_4 = np.frompyfunc(formula_4_func, 2, 1)
formula_12 = np.frompyfunc(formula_12_func, 2, 1)
formula_13 = np.frompyfunc(formula_13_func, 2, 1)
formula_14 = np.frompyfunc(formula_14_func, 3, 1)
optimal = np.frompyfunc(partial(optimal_func, relative_entropy_measure_func), 2, 1)

lower_bounds_4 = compute_boud(formula_4, entropy_values, max_overlap_values)
lower_bounds_12 = compute_boud(formula_12, entropy_values, max_overlap_values)
lower_bounds_13 = compute_boud(formula_13, entropy_values, max_overlap_values)
lower_bounds_14 = compute_boud(formula_14, entropy_values, purity_values, max_overlap_values)
optimal_bounds = compute_boud(optimal, eigenvalue_values, max_overlap_values)

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
axs = axs.ravel()  # type: ignore

# Plot for each entropy value
for i, entropy in enumerate(entropy_values[0]):
    axs[i].plot(max_overlap_values, lower_bounds_4[:, i], marker='x', linestyle='none', color='b', label='Formula 4')  # type: ignore
    axs[i].plot(max_overlap_values, lower_bounds_12[:, i], marker='.', linestyle='none', color='r', label='Formula 12')  # type: ignore
    axs[i].plot(max_overlap_values, lower_bounds_13[:, i], marker='o', linestyle='none', color='y', markerfacecolor='none', label='Formula 13')  # type: ignore
    axs[i].plot(max_overlap_values, lower_bounds_14[:, i], marker='*', linestyle='none', color='m', label='Formula 14')  # type: ignore
    axs[i].plot(max_overlap_values, optimal_bounds[:, i], color='g', label='Optimal')  # type: ignore

    axs[i].set_title(f'Entropy (H) = {entropy:.1f}')  # type: ignore
    axs[i].set_xlabel('Max Basis Overlap (c)')  # type: ignore
    axs[i].set_ylabel('Lower Bound')  # type: ignore
    axs[i].grid(True)  # type: ignore
    axs[i].legend()  # type: ignore
# Adjust layout to prevent overlapping
plt.tight_layout()  # type: ignore
plt.show()  # type: ignore