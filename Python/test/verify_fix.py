"""
Quick test to verify fractional model now responds to weight changes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def ml1_stable(z, alpha, max_terms=100, tol=1e-15):
    """
    Numerically stable Mittag-Leffler function E_α(z)
    Based on the theoretical foundation from the PDF
    """
    if alpha <= 0:
        raise ValueError("Alpha must be positive")

    if abs(z) < tol:
        return 1.0

    # For large negative arguments, use asymptotic behavior
    if z < -50:
        return 0.0

    # Series computation with improved stability
    result = 0.0
    term = 1.0  # First term (n=0)

    for n in range(max_terms):
        if n == 0:
            term = 1.0
        else:
            # Use recurrence relation to avoid overflow
            term *= z / (gamma(alpha * n + 1) / gamma(alpha * (n - 1) + 1))

        if abs(term) < tol:
            break

        result += term

        # Prevent overflow
        if abs(result) > 1e10:
            break

    return result


def fractional_bac_model_corrected(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """
    Theoretically correct fractional BAC model based on PDF theory
    """
    if t == 0:
        return A0, 0.0

    if t < 0:
        return A0, 0.0

    # Stomach concentration A(t)
    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    # Blood concentration B(t)
    if t > 0:
        if abs(k2 - k1) < 1e-10:
            # Special case: k1 ≈ k2
            B_t = A0 * k1 * (t**alpha) * ml2_stable(-k1 * (t**alpha), alpha, alpha + 1)
        else:
            # General case: k1 ≠ k2 (theoretically correct formula)
            term1 = ml1_stable(-k1 * (t**alpha), alpha)
            term2 = ml1_stable(-k2 * (t**beta), beta)
            B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

        # Ensure physical constraints
        B_t = max(0.0, min(B_t, A0))
    else:
        B_t = 0.0

    return max(0.0, A_t), B_t


def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration A0 in stomach"""
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
    tbw_volume = tbw_ratio * weight  # liters
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0


# Test parameters
k1, k2 = 0.8, 1.0
alpha, beta = 0.8, 0.9

print("=== FRACTIONAL MODEL WEIGHT RESPONSE TEST ===")
print()

# Test different weights
weights = [60, 70, 80, 90]
print("Weight | A0 (g/L) | Peak BAC (mg/100mL) at t=2h")
print("-" * 50)

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    _, B_peak = fractional_bac_model_corrected(2.0, A0, k1, k2, alpha, beta)
    B_peak_mg = B_peak * 100

    print(f"{weight:6d} | {A0:8.4f} | {B_peak_mg:18.3f}")

print()
print("If Peak BAC values are different, the fractional model is working correctly!")

# Test recovery times
print("\n=== RECOVERY TIME TEST ===")
print("Weight | Time to reach BAC < 10 mg/100mL")
print("-" * 40)

t_range = np.linspace(0, 15, 500)

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)

    # Find when BAC drops below 10 mg/100mL
    recovery_time = None
    for t in t_range:
        _, B = fractional_bac_model_corrected(t, A0, k1, k2, alpha, beta)
        if B * 100 < 10 and t > 1:  # After initial absorption
            recovery_time = t
            break

    print(f"{weight:6d} | {recovery_time if recovery_time else 'N/A'}")

print()
print(
    "If recovery times are different for different weights, the model is responding correctly!"
)
