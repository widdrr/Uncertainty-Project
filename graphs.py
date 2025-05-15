import numpy as np
from numpy.typing import NDArray
from functools import partial
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar

def entropy(x: np.float64) -> np.float64:
    """
    Computes the binary entropy of a value x.
    Equivalent to the von Neumann entropy for a qubit with eigenvalues p and 1-p.

    Args:
        x (np.float64): The input value.

    Returns:
        np.float64: The computed entropy.
    """

    if np.isclose(x, 0.0) or np.isclose(x, 1.0):
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
    if state_entropy <= 0:
        return (np.float64(1.0), np.float64(1.0))
    if state_entropy >= 1:
        return (np.float64(0.5), np.float64(0.5))


    # Have to numerically solve for the eigenvalue here
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

def save_figure(fig, name: str) -> None:
    """Save a figure to the plots directory, creating it if it doesn't exist."""
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', f'{name}.png'), dpi=300, bbox_inches='tight')

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
        axs[i].plot(max_overlap_values, lower_bounds_4[:, i], marker='x', linestyle='none', color='b', label='Formula 4 (Korzekwa et al.)')  # type: ignore
        axs[i].plot(max_overlap_values, lower_bounds_12[:, i], marker='.', linestyle='none', color='r', label='Formula 12 (Berta et al.)')  # type: ignore
        axs[i].plot(max_overlap_values, lower_bounds_13[:, i], marker='o', linestyle='none', color='y', markerfacecolor='none', label='Formula 13 (Jorge Sanches-Ruiz)')  # type: ignore
        axs[i].plot(max_overlap_values, lower_bounds_14[:, i], marker='*', linestyle='none', color='m', label='Formula 14 (Yuan et al.)')  # type: ignore
        axs[i].plot(max_overlap_values, optimal_bounds[:, i], color='g', label='Optimal')  # type: ignore

        axs[i].set_title(f'Entropy (H) = {entropy:.1f}')  # type: ignore
        axs[i].set_xlabel('Max Basis Overlap (c)')  # type: ignore
        axs[i].set_ylabel('Lower Bound')  # type: ignore
        axs[i].grid(True)  # type: ignore
        axs[i].legend()  # type: ignore
    # Adjust layout to prevent overlapping
    plt.tight_layout()  # type: ignore
    
    # Save the figure
    save_figure(fig, 'lower_bound_plots')
    plt.close(fig)

def plot_heatmap_figure(bounds: list[np.ndarray], titles: list[str], suptitle: str, filename: str, grid_shape: tuple[int, int] = (2, 2)) -> None:
    """Helper function to create and save a heatmap figure.
    
    Args:
        bounds: List of bounds matrices to plot
        titles: List of subplot titles
        suptitle: Main figure title
        filename: Name for saving the figure
        grid_shape: Shape of the subplot grid (rows, cols)
    """
    fig, axs = plt.subplots(*grid_shape, figsize=(15, 10))
    fig.suptitle(suptitle, fontsize=14)
    axs = axs.ravel()

    for idx, (bound, title) in enumerate(zip(bounds, titles)):
        im = axs[idx].imshow(bound, extent=[0.5, 1, 0, 1], cmap='gray', aspect='auto', origin='lower')
        axs[idx].set_title(title)
        plt.colorbar(im, ax=axs[idx])
        axs[idx].set_xlabel('Max Overlap')
        axs[idx].set_ylabel('Entropy')

    # Remove any extra subplots
    for ax in axs[len(bounds):]:
        ax.remove()

    plt.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)

def create_difference_plots(base_bound: np.ndarray, bounds: list[np.ndarray], 
                          base_name: str, other_names: list[str], filename: str) -> None:
    """Helper function to create difference plots between bounds.
    
    Args:
        base_bound: The reference bound to compare against
        bounds: List of bounds to compare with the base bound
        base_name: Name of the base bound
        other_names: Names of the other bounds
        filename: Name for saving the figure
    """
    diffs = [base_bound - bound for bound in bounds]
    titles = [f'{base_name} - {name}' for name in other_names]
    plot_heatmap_figure(diffs, titles, f'Differences with {base_name}', filename)

def heatmaps() -> None:
    max_overlap_values = np.linspace(0.5, 1, 100).reshape(-1,1)
    purity_values = np.linspace(0.5, 1, 100)[::-1].reshape(1,-1)

    values = [entropy_from_purity(purity) for purity in purity_values[0]]
    entropy_values = np.array([value[0] for value in values], dtype=np.float64).reshape(1,-1)
    eigenvalue_values = np.array([value[1] for value in values], dtype=np.float64).reshape(1,-1)

    # Create vectorized formula functions
    formula_4 = np.frompyfunc(formula_4_func, 2, 1)
    formula_12 = np.frompyfunc(formula_12_func, 2, 1)
    formula_13 = np.frompyfunc(formula_13_func, 2, 1)
    formula_14 = np.frompyfunc(formula_14_func, 3, 1)
    optimal = np.frompyfunc(partial(optimal_func, relative_entropy_measure_func), 2, 1)

    # Compute bounds
    lower_bounds_4 = compute_bound(formula_4, entropy_values, max_overlap_values).transpose()
    lower_bounds_12 = compute_bound(formula_12, entropy_values, max_overlap_values).transpose()
    lower_bounds_13 = compute_bound(formula_13, entropy_values, max_overlap_values).transpose()
    lower_bounds_14 = compute_bound(formula_14, entropy_values, purity_values, max_overlap_values).transpose()
    optimal_bounds = compute_bound(optimal, eigenvalue_values, max_overlap_values).transpose()    
    bounds = [lower_bounds_4, lower_bounds_12, lower_bounds_13, lower_bounds_14, optimal_bounds]
    bound_names = ['Formula 4 (Korzekwa et al.)', 'Formula 12 (Berta et al.)', 'Formula 13 (Jorge Sanches-Ruiz)', 'Formula 14 (Yuan et al.)', 'Optimal']

    # Plot all bounds
    plot_heatmap_figure(bounds, bound_names, 'Lower Bounds', 'lower_bound_heatmaps', (2, 3))

    # Create difference plots for each formula
    other_bounds = [b for b in bounds if b is not lower_bounds_4]
    other_names = [n for n in bound_names if n != 'Formula 4']
    create_difference_plots(lower_bounds_4, other_bounds, 'F4', other_names, 'f4_differences')

    other_bounds = [b for b in bounds if b is not lower_bounds_12]
    other_names = [n for n in bound_names if n != 'Formula 12']
    create_difference_plots(lower_bounds_12, other_bounds, 'F12', other_names, 'f12_differences')

    other_bounds = [b for b in bounds if b is not lower_bounds_13]
    other_names = [n for n in bound_names if n != 'Formula 13']
    create_difference_plots(lower_bounds_13, other_bounds, 'F13', other_names, 'f13_differences')

    other_bounds = [b for b in bounds if b is not lower_bounds_14]
    other_names = [n for n in bound_names if n != 'Formula 14']
    create_difference_plots(lower_bounds_14, other_bounds, 'F14', other_names, 'f14_differences')

    other_bounds = [b for b in bounds if b is not optimal_bounds]
    other_names = [n for n in bound_names if n != 'Optimal']
    create_difference_plots(optimal_bounds, other_bounds, 'Optimal', other_names, 'optimal_differences')

def main(mode: str | None = None) -> None:
    """Run both plots and heatmaps."""
    plots()
    heatmaps()


if __name__ == "__main__":
    main()