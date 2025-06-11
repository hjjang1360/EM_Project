"""
Debug the fractional model recovery time issue
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

print("=== DEBUGGING FRACTIONAL MODEL RECOVERY TIME ISSUE ===")
print()

# Test 1: Check if different weights give different A0 values
weights = [60, 70, 80, 90]
print("Weight vs A0 test:")
for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    print(f"Weight {weight}kg: A0 = {A0:.6f} g/L")

print()

# Test 2: Check BAC curves for different weights
print("=== BAC CURVES FOR DIFFERENT WEIGHTS ===")
t_test = np.array([0, 1, 2, 4, 6, 8, 10, 12])

for weight in [60, 80]:  # Test two different weights
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    print(f"\nWeight {weight}kg (A0={A0:.6f}):")
    print("Time | Classical BAC | Fractional BAC")
    print("-" * 40)

    for t in t_test:
        _, B_c = classical_bac_model(t, A0, k1, k2)
        _, B_f = fractional_bac_model_corrected(t, A0, k1, k2, alpha, beta)
        print(f"{t:4.0f} | {B_c*100:11.3f} | {B_f*100:12.3f}")

print()

# Test 3: Check recovery time calculation step by step
print("=== STEP-BY-STEP RECOVERY TIME CALCULATION ===")

t_extended = np.linspace(0, 15, 200)
weights_test = [60, 80]

for weight in weights_test:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    print(f"\nWeight {weight}kg (A0={A0:.6f}):")

    # Generate BAC curve
    bac_values = []
    for t in t_extended:
        _, B = fractional_bac_model_corrected(t, A0, k1, k2, alpha, beta)
        bac_values.append(B * 100)  # Convert to mg/100mL

    bac_array = np.array(bac_values)

    # Check max BAC
    max_bac = np.max(bac_array)
    max_idx = np.argmax(bac_array)
    max_time = t_extended[max_idx]

    print(f"  Max BAC: {max_bac:.3f} mg/100mL at t={max_time:.2f}h")

    # Check if BAC ever goes below 10 mg/100mL
    below_10 = np.where(bac_array < 10)[0]
    if len(below_10) > 0:
        # Find first time after peak where BAC < 10
        after_peak = below_10[below_10 > max_idx]
        if len(after_peak) > 0:
            recovery_time = t_extended[after_peak[0]]
            print(f"  Recovery time: {recovery_time:.2f}h")
        else:
            print("  Never recovers after peak")
    else:
        print("  Never goes below 10 mg/100mL")

    # Use find_threshold_times function
    _, t_f = find_threshold_times(t_extended, bac_array)
    print(f"  find_threshold_times result: {t_f}")

print()

# Test 4: Check if the issue is in find_threshold_times function
print("=== TESTING find_threshold_times FUNCTION ===")

# Create test data
test_bac = np.array([0, 5, 15, 25, 20, 15, 8, 5, 3, 1])  # mg/100mL
test_time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("Test BAC curve:", test_bac)
print("Test time:", test_time)

t_i, t_f = find_threshold_times(
    test_time, test_bac, threshold_high=0.08, threshold_low=0.01
)
print(f"find_threshold_times result: t_i={t_i}, t_f={t_f}")

# Check the logic inside find_threshold_times
print("\nDebug find_threshold_times logic:")
if np.max(test_bac) > 1:  # This will be True since max is 25
    bac_values = test_bac / 100  # Convert to g/100mL
    print(f"Converted BAC values: {bac_values}")
else:
    bac_values = test_bac

above_low = np.where(bac_values > 0.01)[0]  # 10 mg/100mL = 0.01 g/100mL
print(f"Indices above 10 mg/100mL: {above_low}")
if len(above_low) > 0:
    t_f_manual = test_time[above_low[-1]]
    print(f"Manual calculation: t_f = {t_f_manual}")
