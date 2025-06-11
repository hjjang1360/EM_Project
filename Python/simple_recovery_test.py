#!/usr/bin/env python3
"""
Simple validation of the recovery time fix
"""

import numpy as np
from scipy.special import gamma


def ml1_stable(z, alpha, max_terms=100, tol=1e-15):
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if abs(z) < tol:
        return 1.0
    if z < -50:
        return 0.0

    result = 0.0
    term = 1.0

    for n in range(max_terms):
        if n == 0:
            term = 1.0
        else:
            term *= z / (gamma(alpha * n + 1) / gamma(alpha * (n - 1) + 1))

        if abs(term) < tol:
            break

        result += term

        if abs(result) > 1e10:
            break

    return result


def fractional_bac_model(t, A0, k1=0.8, k2=1.0, alpha=0.8, beta=0.9):
    if t <= 0:
        return A0, 0.0

    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    if abs(k2 - k1) < 1e-10:
        B_t = A0 * k1 * (t**alpha) * ml1_stable(-k1 * (t**alpha), alpha)
    else:
        term1 = ml1_stable(-k1 * (t**alpha), alpha)
        term2 = ml1_stable(-k2 * (t**beta), beta)
        B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

    B_t = max(0.0, min(B_t, A0))
    return max(0.0, A_t), B_t


def find_recovery_times_old(t_array, bac_array):
    """OLD problematic method"""
    bac_mg = bac_array * 100

    legal_idx = np.where(bac_mg <= 50)[0]
    legal_time = t_array[legal_idx[0]] if len(legal_idx) > 0 else None

    safe_idx = np.where(bac_mg <= 30)[0]
    safe_time = t_array[safe_idx[0]] if len(safe_idx) > 0 else None

    recovery_idx = np.where(bac_mg <= 10)[0]
    recovery_time = t_array[recovery_idx[0]] if len(recovery_idx) > 0 else None

    return legal_time, safe_time, recovery_time


def find_recovery_times_new(t_array, bac_array):
    """NEW fixed method"""
    bac_mg = bac_array * 100

    # Find peak first
    peak_idx = np.argmax(bac_mg)
    peak_time = t_array[peak_idx]

    # Only consider post-peak
    post_peak_mask = t_array > peak_time
    post_peak_times = t_array[post_peak_mask]
    post_peak_bac = bac_mg[post_peak_mask]

    legal_idx = np.where(post_peak_bac <= 50)[0]
    legal_time = post_peak_times[legal_idx[0]] if len(legal_idx) > 0 else None

    safe_idx = np.where(post_peak_bac <= 30)[0]
    safe_time = post_peak_times[safe_idx[0]] if len(safe_idx) > 0 else None

    recovery_idx = np.where(post_peak_bac <= 10)[0]
    recovery_time = post_peak_times[recovery_idx[0]] if len(recovery_idx) > 0 else None

    return legal_time, safe_time, recovery_time


def main():
    print("Recovery Time Logic Test")
    print("=" * 40)

    # Test case: Male, 25y, 70kg, Soju 360mL
    weight = 70
    tbw_ratio = 0.68
    volume = 360
    abv = 17

    # Calculate A0
    rho_ethanol = 0.789
    alcohol_mass = volume * (abv / 100) * rho_ethanol
    tbw_volume = tbw_ratio * weight
    A0 = alcohol_mass / tbw_volume

    print(f"Test scenario: Male, 25y, 70kg, Soju 360mL 17%")
    print(f"A0 = {A0:.3f} g/L")
    print(f"Alcohol mass = {alcohol_mass:.1f}g")

    # Calculate BAC over time
    t_hours = np.linspace(0, 24, 1000)
    bac_values = []

    for t in t_hours:
        _, B_t = fractional_bac_model(t, A0)
        bac_values.append(B_t)

    bac_array = np.array(bac_values)
    bac_mg = bac_array * 100

    # Find peak
    peak_idx = np.argmax(bac_mg)
    peak_time = t_hours[peak_idx]
    peak_bac = bac_mg[peak_idx]

    print(f"\nPeak BAC: {peak_bac:.1f} mg/100mL at {peak_time:.1f} hours")

    # Test old method
    old_legal, old_safe, old_recovery = find_recovery_times_old(t_hours, bac_array)
    print(f"\nOLD METHOD (problematic):")
    print(f"  Legal (50mg): {old_legal:.1f}h" if old_legal else "  Legal: None")
    print(f"  Safe (30mg):  {old_safe:.1f}h" if old_safe else "  Safe: None")
    print(f"  Full (10mg):  {old_recovery:.1f}h" if old_recovery else "  Full: None")

    # Test new method
    new_legal, new_safe, new_recovery = find_recovery_times_new(t_hours, bac_array)
    print(f"\nNEW METHOD (fixed):")
    print(f"  Legal (50mg): {new_legal:.1f}h" if new_legal else "  Legal: None")
    print(f"  Safe (30mg):  {new_safe:.1f}h" if new_safe else "  Safe: None")
    print(f"  Full (10mg):  {new_recovery:.1f}h" if new_recovery else "  Full: None")

    # Analysis
    print(f"\nANALYSIS:")
    print(f"The old method likely finds t=0 where BAC=0")
    print(f"The new method correctly finds recovery AFTER peak")

    # Check what happens at early times
    early_bac = bac_mg[:10]  # First 10 time points
    print(f"\nEarly BAC values (first 10 points):")
    for i in range(10):
        print(f"  t={t_hours[i]:.3f}h: {early_bac[i]:.1f} mg/100mL")

    print(f"\n✅ Test complete!")
    print(f"🔧 The fix ensures we only look AFTER the peak time.")


if __name__ == "__main__":
    main()
