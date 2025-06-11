"""
Direct test of the comprehensive plot issue
"""

import numpy as np
import sys

sys.path.append(r"d:\Kentech\학습 관련 자료\2-1\공업수학 1\EM_Project\Python")

from plot_corrected_comprehensive import (
    fractional_bac_model_corrected,
    classical_bac_model,
    calculate_initial_concentration,
    find_threshold_times,
)

# Parameters
k1, k2 = 0.8, 1.0
alpha, beta = 0.8, 0.9

print("=== DIRECT COMPREHENSIVE PLOT DEBUG ===")
print()

# Test TBW scenario exactly as in the plot
tbw_values = np.linspace(30, 70, 5)  # Smaller range for testing
t_extended = np.linspace(0, 15, 500)

print("TBW | Weight | A0      | Recovery (Classical) | Recovery (Fractional)")
print("-" * 70)

for tbw in tbw_values:
    # Male scenario (beer) exactly as in code
    weight_male = tbw / 0.68
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)

    # Classical recovery time
    bac_male_c = []
    for time_point in t_extended:
        _, B = classical_bac_model(time_point, A0_male, k1, k2)
        bac_male_c.append(B * 100)  # Convert to mg/100mL

    _, t_f_male_c = find_threshold_times(t_extended, np.array(bac_male_c))

    # Fractional recovery time
    bac_male_f = []
    for time_point in t_extended:
        _, B = fractional_bac_model_corrected(time_point, A0_male, k1, k2, alpha, beta)
        bac_male_f.append(B * 100)

    _, t_f_male_f = find_threshold_times(t_extended, np.array(bac_male_f))

    print(
        f"{tbw:4.1f} | {weight_male:6.1f} | {A0_male:.4f} | {t_f_male_c if t_f_male_c else 'N/A':18} | {t_f_male_f if t_f_male_f else 'N/A'}"
    )

print()

# Check if fractional model values are reasonable
print("=== DETAILED FRACTIONAL MODEL CHECK ===")
A0_test = calculate_initial_concentration(70, 0.68, 350, 5)
print(f"Test A0 = {A0_test:.4f} g/L (70kg male, beer)")
print()

t_test = np.array([0, 1, 2, 4, 6, 8, 10, 12])
print("Time | Classical | Fractional | Max BAC so far")
print("-" * 45)

max_bac_c = 0
max_bac_f = 0

for t in t_test:
    _, B_c = classical_bac_model(t, A0_test, k1, k2)
    _, B_f = fractional_bac_model_corrected(t, A0_test, k1, k2, alpha, beta)

    B_c_mg = B_c * 100
    B_f_mg = B_f * 100

    max_bac_c = max(max_bac_c, B_c_mg)
    max_bac_f = max(max_bac_f, B_f_mg)

    print(
        f"{t:4.0f} | {B_c_mg:9.3f} | {B_f_mg:10.3f} | C:{max_bac_c:.1f} F:{max_bac_f:.1f}"
    )

print()
print(
    "If all fractional values are very small or constant, the issue is in the Mittag-Leffler function."
)
