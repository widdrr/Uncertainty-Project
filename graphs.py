import numpy as np
from numpy.typing import NDArray
from functools import partial
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar
import argparse

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
        Tuple[np.float64, np.float64]: A tuple containing the purity and the eigenvalue.
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

def entropy_from_purity(state_purity: np.float64) -> Tuple[np.float64, np.float64]:
    """
    Computes the entropy of a qubit given its purity.

    For a qubit with eigenvalues p and 1-p, the purity is:
        P = p^2 + (1-p)^2
    and the von Neumann entropy is:
        S = -p*log2(p) - (1-p)*log2(1-p)

    Args:
        state_purity (np.float64): The purity of the state.

    Returns:
        Tuple[np.float64, np.float64]: A tuple containing the entropy and the eigenvalue.
    """
    if state_purity >= 1:
        return (np.float64(0.0), np.float64(1.0))
    if state_purity <= 0.5:
        return (np.float64(1.0), np.float64(0.5))

    p = (1 + np.sqrt(2 * state_purity - 1)) / 2
    return (entropy(p), np.float64(p))


def compute_bound(formula: np.ufunc, *args: np.ndarray) -> np.ndarray:
    """
    Compute the lower bound of uncertainty using a vectorized formula, applied elementwise on the vectorized inputs.
    
    Args:
        formula (np.ufunc): The vectorized formula to use for the lower bound.
        *args (np.ndarray): Vectors with the arguments for formula.
    
    Returns:
        np.ndarray: An array of computed lower bounds.
    """
    
    return np.clip(formula(*args), 0, None).astype(np.float64)

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

def plots() -> None:
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

    lower_bounds_4 = compute_bound(formula_4, entropy_values, max_overlap_values)
    lower_bounds_12 = compute_bound(formula_12, entropy_values, max_overlap_values)
    lower_bounds_13 = compute_bound(formula_13, entropy_values, max_overlap_values)
    lower_bounds_14 = compute_bound(formula_14, entropy_values, purity_values, max_overlap_values)
    optimal_bounds = compute_bound(optimal, eigenvalue_values, max_overlap_values)

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

def heatmaps() -> None:
    max_overlap_values = np.linspace(0.5, 1, 100).reshape(-1,1)
    purity_values = np.linspace(0.5, 1, 100)[::-1].reshape(1,-1)

    values = [entropy_from_purity(purity) for purity in purity_values[0]]
    entropy_values = np.array([value[0] for value in values], dtype=np.float64).reshape(1,-1)
    eigenvalue_values = np.array([value[1] for value in values], dtype=np.float64).reshape(1,-1)

    formula_4 = np.frompyfunc(formula_4_func, 2, 1)
    formula_12 = np.frompyfunc(formula_12_func, 2, 1)
    formula_13 = np.frompyfunc(formula_13_func, 2, 1)
    formula_14 = np.frompyfunc(formula_14_func, 3, 1)
    optimal = np.frompyfunc(partial(optimal_func, relative_entropy_measure_func), 2, 1)

    lower_bounds_4 = compute_bound(formula_4, entropy_values, max_overlap_values).transpose()
    lower_bounds_12 = compute_bound(formula_12, entropy_values, max_overlap_values).transpose()
    lower_bounds_13 = compute_bound(formula_13, entropy_values, max_overlap_values).transpose()
    lower_bounds_14 = compute_bound(formula_14, entropy_values, purity_values, max_overlap_values).transpose()
    optimal_bounds = compute_bound(optimal, eigenvalue_values, max_overlap_values).transpose()

    fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Lower Bounds', fontsize=14)
    axs1 = axs1.ravel()

    # Plot all bounds
    bounds_list = [lower_bounds_4, lower_bounds_12, lower_bounds_13, 
                   lower_bounds_14, optimal_bounds]
    titles = ['Formula 4', 'Formula 12', 'Formula 13', 'Formula 14', 'Optimal']
    
    for idx, (bound, title) in enumerate(zip(bounds_list, titles)):
        im = axs1[idx].imshow(bound, extent=[0.5, 1, 0, 1], cmap='gray', 
                             aspect='auto', origin='lower')
        axs1[idx].set_title(title)
        plt.colorbar(im, ax=axs1[idx])
        axs1[idx].set_xlabel('Max Overlap')
        axs1[idx].set_ylabel('Entropy')

    # Remove the extra subplot
    axs1[-1].remove()
    plt.tight_layout()

    # Figure 2: F4 differences
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))
    fig2.suptitle('Differences with Formula 4', fontsize=14)
    axs2 = axs2.ravel()

    diffs_f4 = [lower_bounds_4 - lower_bounds_12, lower_bounds_4 - lower_bounds_13,
                lower_bounds_4 - lower_bounds_14, lower_bounds_4 - optimal_bounds]
    titles_f4 = ['F4 - F12', 'F4 - F13', 'F4 - F14', 'F4 - Optimal']

    for ax, diff, title in zip(axs2, diffs_f4, titles_f4):
        im = ax.imshow(diff, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Max Overlap')
        ax.set_ylabel('Entropy')

    plt.tight_layout()

    # Figure 3: F12 differences
    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 10))
    fig3.suptitle('Differences with Formula 12', fontsize=14)
    axs3 = axs3.ravel()

    diffs_f12 = [lower_bounds_12 - lower_bounds_4, lower_bounds_12 - lower_bounds_13,
                 lower_bounds_12 - lower_bounds_14, lower_bounds_12 - optimal_bounds]
    titles_f12 = ['F12 - F4', 'F12 - F13', 'F12 - F14', 'F12 - Optimal']

    for ax, diff, title in zip(axs3, diffs_f12, titles_f12):
        im = ax.imshow(diff, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Max Overlap')
        ax.set_ylabel('Entropy')

    plt.tight_layout()

    # Figure 4: F13 differences
    fig4, axs4 = plt.subplots(2, 2, figsize=(10, 10))
    fig4.suptitle('Differences with Formula 13', fontsize=14)
    axs4 = axs4.ravel()

    diffs_f13 = [lower_bounds_13 - lower_bounds_4, lower_bounds_13 - lower_bounds_12,
                 lower_bounds_13 - lower_bounds_14, lower_bounds_13 - optimal_bounds]
    titles_f13 = ['F13 - F4', 'F13 - F12', 'F13 - F14', 'F13 - Optimal']

    for ax, diff, title in zip(axs4, diffs_f13, titles_f13):
        im = ax.imshow(diff, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Max Overlap')
        ax.set_ylabel('Entropy')

    plt.tight_layout()

    # Figure 5: F14 differences
    fig5, axs5 = plt.subplots(2, 2, figsize=(10, 10))
    fig5.suptitle('Differences with Formula 14', fontsize=14)
    axs5 = axs5.ravel()

    diffs_f14 = [lower_bounds_14 - lower_bounds_4, lower_bounds_14 - lower_bounds_12,
                 lower_bounds_14 - lower_bounds_13, lower_bounds_14 - optimal_bounds]
    titles_f14 = ['F14 - F4', 'F14 - F12', 'F14 - F13', 'F14 - Optimal']

    for ax, diff, title in zip(axs5, diffs_f14, titles_f14):
        im = ax.imshow(diff, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Max Overlap')
        ax.set_ylabel('Entropy')

    plt.tight_layout()

    # Figure 6: Optimal differences
    fig6, axs6 = plt.subplots(2, 2, figsize=(10, 10))
    fig6.suptitle('Differences with Optimal', fontsize=14)
    axs6 = axs6.ravel()

    diffs_opt = [optimal_bounds - lower_bounds_4, optimal_bounds - lower_bounds_12,
                 optimal_bounds - lower_bounds_13, optimal_bounds - lower_bounds_14]
    titles_opt = ['Optimal - F4', 'Optimal - F12', 'Optimal - F13', 'Optimal - F14']

    for ax, diff, title in zip(axs6, diffs_opt, titles_opt):
        im = ax.imshow(diff, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Max Overlap')
        ax.set_ylabel('Entropy')

    plt.tight_layout()
    plt.show()


def main(mode: str) -> None:
    if mode == "plots":
        plots()
    elif mode in ["heatmaps","heatmaps-f4","heatmaps-f12","heatmaps-f13","heatmaps-f14,heatmaps-o"] :
        heatmaps()
    else:
        print(f"Unknown mode: {mode}. Please use 'plots' or 'heatmap'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphs")
    parser.add_argument("mode", type=str, help="Mode of operation: 'plot' or 'heatmaps'")
    args = parser.parse_args()
    main(args.mode)