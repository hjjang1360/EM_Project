"""
Direct test of Mittag-Leffler functions
"""

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


def ml1_stable(z, alpha, max_terms=50, tol=1e-12):
    """
    Numerically stable Mittag-Leffler function E_α(z)
    """
    if alpha <= 0:
        raise ValueError("Alpha must be positive")

    if abs(z) < tol:
        return 1.0

    # For large negative arguments, use asymptotic behavior
    if z < -30:
        return 0.0

    # For large positive arguments, prevent overflow
    if z > 30:
        return np.inf

    result = 0.0
    for n in range(max_terms):
        try:
            term = (z**n) / gamma(alpha * n + 1)
            if abs(term) < tol:
                break
            result += term

            # Prevent overflow
            if abs(result) > 1e8:
                break
        except (OverflowError, ValueError):
            break

    return result


# Test Mittag-Leffler function
print("=== MITTAG-LEFFLER FUNCTION TEST ===")
print()

alpha = 0.8
z_values = [-0.5, -1.0, -2.0, -3.0, -4.0]

print("z     | E_α(z)")
print("-" * 20)
for z in z_values:
    ml_val = ml1_stable(z, alpha)
    print(f"{z:5.1f} | {ml_val:.6f}")

print()

# Test fractional BAC model directly
k1, k2 = 0.8, 1.0
alpha, beta = 0.8, 0.9


def test_fractional_bac(t, A0):
    """Simplified test"""
    if t == 0:
        return A0, 0.0

    # Stomach concentration A(t)
    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    # Blood concentration B(t) - simplified
    term1 = ml1_stable(-k1 * (t**alpha), alpha)
    term2 = ml1_stable(-k2 * (t**beta), beta)
    B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

    return A_t, B_t


print("=== FRACTIONAL BAC TEST ===")
print()

# Test with different A0 values
A0_values = [0.1, 0.2, 0.3, 0.4]  # Different initial concentrations
t_test = 2.0

print("A0    | A(t=2) | B(t=2)")
print("-" * 25)
for A0 in A0_values:
    A_t, B_t = test_fractional_bac(t_test, A0)
    print(f"{A0:.1f} | {A_t:.4f} | {B_t:.4f}")

print()
print(
    "The problem might be in find_threshold_times function or the Mittag-Leffler computation."
)
print("Let's check if BAC values are actually different...")

# Check BAC progression for different A0
print("\n=== BAC PROGRESSION TEST ===")
A0_small = 0.2
A0_large = 0.4

t_range = np.array([0, 1, 2, 4, 6, 8, 10])

print("Time | Small A0 | Large A0 | Difference")
print("-" * 40)
for t in t_range:
    _, B_small = test_fractional_bac(t, A0_small)
    _, B_large = test_fractional_bac(t, A0_large)
    diff = B_large - B_small
    print(f"{t:4.0f} | {B_small*100:8.3f} | {B_large*100:8.3f} | {diff*100:10.3f}")
