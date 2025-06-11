"""
Debug fractional model behavior
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

# Test parameters
k1, k2 = 0.8, 1.0
alpha, beta = 0.8, 0.9

print("=== DEBUGGING FRACTIONAL MODEL BEHAVIOR ===")
print()

# Test 1: Different weights should give different A0 values
weights = [60, 70, 80, 90]
print("Weight vs A0 test:")
for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    print(f"Weight {weight}kg: A0 = {A0:.4f} g/L")

print()

# Test 2: Check if fractional model gives different BAC for different A0
print("BAC comparison at t=2h:")
print("Weight | A0      | Classical BAC | Fractional BAC")
print("-" * 50)

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)

    _, B_c = classical_bac_model(2.0, A0, k1, k2)
    _, B_f = fractional_bac_model_corrected(2.0, A0, k1, k2, alpha, beta)

    print(f"{weight:6d} | {A0:.4f} | {B_c*100:11.3f} | {B_f*100:12.3f}")

print()

# Test 3: Check recovery times for different weights
print("Recovery time test (time to BAC < 10 mg/100mL):")
print("Weight | Classical Recovery | Fractional Recovery")
print("-" * 50)

t_extended = np.linspace(0, 15, 500)

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)

    # Classical recovery time
    bac_c = []
    for t in t_extended:
        _, B = classical_bac_model(t, A0, k1, k2)
        bac_c.append(B * 100)

    _, t_f_c = find_threshold_times(t_extended, np.array(bac_c))

    # Fractional recovery time
    bac_f = []
    for t in t_extended:
        _, B = fractional_bac_model_corrected(t, A0, k1, k2, alpha, beta)
        bac_f.append(B * 100)

    _, t_f_f = find_threshold_times(t_extended, np.array(bac_f))

    print(f"{weight:6d} | {t_f_c if t_f_c else 'N/A':17} | {t_f_f if t_f_f else 'N/A'}")

print()

# Test 4: Check if fractional model is stuck at some value
print("Detailed fractional model values over time (70kg):")
A0_test = calculate_initial_concentration(70, 0.68, 350, 5)
print(f"A0 = {A0_test:.4f} g/L")
print()
print("Time | Fractional BAC")
print("-" * 25)

test_times = [0, 0.5, 1, 2, 4, 6, 8, 10, 12]
for t in test_times:
    _, B_f = fractional_bac_model_corrected(t, A0_test, k1, k2, alpha, beta)
    print(f"{t:4.1f} | {B_f*100:12.3f}")
