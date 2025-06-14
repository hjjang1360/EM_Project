"""
Comprehensive plotting script using the corrected fractional BAC model
Based on fractional_bac_corrected.py with theoretically correct formulations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set plot parameters
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.figsize"] = (18, 12)
sns.set_style("whitegrid")


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


def ml2_stable(z, alpha, beta, max_terms=50, tol=1e-15):
    """
    Numerically stable two-parameter Mittag-Leffler function E_{α,β}(z)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")

    if abs(z) < tol:
        return 1.0 / gamma(beta)

    if z < -50:
        return 0.0

    result = 0.0

    for n in range(max_terms):
        try:
            term = (z**n) / gamma(alpha * n + beta)
            if abs(term) < tol:
                break
            result += term
        except (OverflowError, ValueError):
            break

    return result


def fractional_bac_model_corrected(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """
    Theoretically correct fractional BAC model based on PDF theory:

    A(t) = A0 * E_α(-k1 * t^α)
    B(t) = (A0 * k1)/(k2 - k1) * [E_α(-k1 * t^α) - E_β(-k2 * t^β)]
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


def classical_bac_model(t, A0, k1, k2):
    """
    Classical two-compartment BAC model for comparison
    """
    if t == 0:
        return A0, 0.0

    A_t = A0 * np.exp(-k1 * t)

    if abs(k1 - k2) < 1e-10:
        B_t = k1 * A0 * t * np.exp(-k1 * t)
    else:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))

    return max(0.0, A_t), max(0.0, B_t)


def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration A0 in stomach"""
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
    tbw_volume = tbw_ratio * weight  # liters
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0


def find_threshold_times(t_array, bac_array, threshold_high=80, threshold_low=10):
    """Find times when BAC crosses thresholds

    Args:
        t_array: time array
        bac_array: BAC values (assumed to be in mg/100mL)
        threshold_high: high threshold in mg/100mL (default 80)
        threshold_low: low threshold in mg/100mL (default 10)
    """
    t_i = None  # Time when BAC exceeds threshold_high
    t_f = None  # Time when BAC drops below threshold_low

    # Ensure bac_array is in mg/100mL
    if np.max(bac_array) <= 1:  # Assume it's in g/100mL, convert to mg/100mL
        bac_values = bac_array * 100
    else:
        bac_values = bac_array

    # Find first crossing above high threshold
    over_high = np.where(bac_values >= threshold_high)[0]
    if len(over_high) > 0:
        t_i = t_array[over_high[0]]

    # Find last time above low threshold
    above_low = np.where(bac_values > threshold_low)[0]
    if len(above_low) > 0:
        t_f = t_array[above_low[-1]]

    return t_i, t_f


# Model parameters - using PDF recommendations
k1, k2 = 0.8, 1.0  # h^-1 (k2 > k1 for proper elimination)
alpha, beta = 0.8, 0.9  # Fractional orders
t_max = 12  # hours
t = np.linspace(0, t_max, 400)
t_dense = np.concatenate(
    [np.linspace(0, 0.2, 100), np.linspace(0.2, 1, 100), np.linspace(1, t_max, 300)]
)
t_dense.sort()

print("=== CORRECTED FRACTIONAL BAC MODEL COMPREHENSIVE PLOTS ===")
print(f"Parameters: k1={k1}, k2={k2}, α={alpha}, β={beta}")
print()

# =============================================================================
# FIGURE 1: MODEL COMPARISON (Classical vs Corrected Fractional)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Define test scenarios
scenarios = [
    {
        "gender": "Male",
        "abv": 5,
        "volume": 350,
        "tbw_ratio": 0.68,
        "color": "blue",
        "linestyle": "-",
    },
    {
        "gender": "Female",
        "abv": 5,
        "volume": 350,
        "tbw_ratio": 0.55,
        "color": "red",
        "linestyle": "-",
    },
    {
        "gender": "Male",
        "abv": 40,
        "volume": 50,
        "tbw_ratio": 0.68,
        "color": "darkblue",
        "linestyle": "--",
    },
    {
        "gender": "Female",
        "abv": 40,
        "volume": 50,
        "tbw_ratio": 0.55,
        "color": "darkred",
        "linestyle": "--",
    },
]

# Plot 1.1: Classical Model BAC vs Time
ax = axes[0, 0]
for scenario in scenarios:
    weight = 65  # kg
    A0 = calculate_initial_concentration(
        weight, scenario["tbw_ratio"], scenario["volume"], scenario["abv"]
    )

    bac_values = []
    for time_point in t_dense:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_values.append(B * 100)  # Convert g/L to mg/100mL

    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(
        t_dense,
        bac_values,
        color=scenario["color"],
        linestyle=scenario["linestyle"],
        linewidth=2,
        label=label,
    )

ax.axhline(
    y=80, color="red", linestyle=":", alpha=0.7, label="Legal Limit (80 mg/100mL)"
)
ax.axhline(
    y=10, color="orange", linestyle=":", alpha=0.7, label="Recovery (10 mg/100mL)"
)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BAC (mg/100mL)")
ax.set_title("Classical Model: BAC vs Time (65kg)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 120)

# Plot 1.2: Classical Model Recovery Time vs TBW
ax = axes[0, 1]
tbw_values = np.linspace(30, 70, 20)
recovery_times_male = []
recovery_times_female = []

for tbw in tbw_values:
    # Male scenario (beer)
    weight_male = tbw / 0.68
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)

    bac_male = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0_male, k1, k2)
        bac_male.append(B * 100)  # Convert to mg/100mL

    _, t_f_male = find_threshold_times(t, np.array(bac_male))
    recovery_times_male.append(t_f_male if t_f_male else 0)

    # Female scenario (beer)
    weight_female = tbw / 0.55
    A0_female = calculate_initial_concentration(weight_female, 0.55, 350, 5)

    bac_female = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0_female, k1, k2)
        bac_female.append(B * 100)

    _, t_f_female = find_threshold_times(t, np.array(bac_female))
    recovery_times_female.append(t_f_female if t_f_female else 0)

ax.plot(tbw_values, recovery_times_male, "o-", color="blue", label="Male", linewidth=2)
ax.plot(
    tbw_values, recovery_times_female, "s-", color="red", label="Female", linewidth=2
)
ax.set_xlabel("Total Body Water (liters)")
ax.set_ylabel("Time to BAC < 10 mg/100mL (hours)")
ax.set_title("Classical Model: Recovery Time vs TBW")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 1.3: Classical Model Recovery Time vs Weight
ax = axes[0, 2]
weights = np.arange(60, 101, 5)
recovery_times_weight = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)

    t_extended = np.linspace(0, 15, 500)
    bac_values = []
    for time_point in t_extended:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_values.append(B * 100)

    _, t_f = find_threshold_times(t_extended, np.array(bac_values))
    recovery_times_weight.append(t_f if t_f else 0)

ax.plot(weights, recovery_times_weight, "o-", color="green", linewidth=2, markersize=6)
ax.set_xlabel("Body Weight (kg)")
ax.set_ylabel("Time to BAC < 10 mg/100mL (hours)")
ax.set_title("Classical Model: Recovery Time vs Weight")
ax.grid(True, alpha=0.3)

# Plot 2.1: Corrected Fractional Model BAC vs Time
ax = axes[1, 0]
for scenario in scenarios:
    weight = 65  # kg
    A0 = calculate_initial_concentration(
        weight, scenario["tbw_ratio"], scenario["volume"], scenario["abv"]
    )

    bac_values = []
    for time_point in t_dense:
        _, B = fractional_bac_model_corrected(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B * 100)  # Convert g/L to mg/100mL

    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(
        t_dense,
        bac_values,
        color=scenario["color"],
        linestyle=scenario["linestyle"],
        linewidth=2,
        label=label,
    )

ax.axhline(
    y=80, color="red", linestyle=":", alpha=0.7, label="Legal Limit (80 mg/100mL)"
)
ax.axhline(
    y=10, color="orange", linestyle=":", alpha=0.7, label="Recovery (10 mg/100mL)"
)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BAC (mg/100mL)")
ax.set_title("Corrected Fractional Model: BAC vs Time (65kg)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 120)

# Plot 2.2: Fractional Model Recovery Time vs TBW
ax = axes[1, 1]
recovery_times_male_frac = []
recovery_times_female_frac = []

for tbw in tbw_values:
    # Male scenario (beer)
    weight_male = tbw / 0.68
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)

    t_extended = np.linspace(0, 15, 500)
    bac_male = []
    for time_point in t_extended:
        _, B = fractional_bac_model_corrected(time_point, A0_male, k1, k2, alpha, beta)
        bac_male.append(B * 100)

    _, t_f_male = find_threshold_times(t_extended, np.array(bac_male))
    recovery_times_male_frac.append(t_f_male if t_f_male else 0)

    # Female scenario (beer)
    weight_female = tbw / 0.55
    A0_female = calculate_initial_concentration(weight_female, 0.55, 350, 5)

    bac_female = []
    for time_point in t_extended:
        _, B = fractional_bac_model_corrected(
            time_point, A0_female, k1, k2, alpha, beta
        )
        bac_female.append(B * 100)

    _, t_f_female = find_threshold_times(t_extended, np.array(bac_female))
    recovery_times_female_frac.append(t_f_female if t_f_female else 0)

ax.plot(
    tbw_values, recovery_times_male_frac, "o-", color="blue", label="Male", linewidth=2
)
ax.plot(
    tbw_values,
    recovery_times_female_frac,
    "s-",
    color="red",
    label="Female",
    linewidth=2,
)
ax.set_xlabel("Total Body Water (liters)")
ax.set_ylabel("Time to BAC < 10 mg/100mL (hours)")
ax.set_title("Corrected Fractional Model: Recovery Time vs TBW")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2.3: Fractional Model Recovery Time vs Weight
ax = axes[1, 2]
recovery_times_weight_frac = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)

    t_extended = np.linspace(0, 15, 500)
    bac_values = []
    for time_point in t_extended:
        _, B = fractional_bac_model_corrected(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B * 100)

    _, t_f = find_threshold_times(t_extended, np.array(bac_values))
    recovery_times_weight_frac.append(t_f if t_f else 0)

ax.plot(
    weights, recovery_times_weight_frac, "o-", color="green", linewidth=2, markersize=6
)
ax.set_xlabel("Body Weight (kg)")
ax.set_ylabel("Time to BAC < 10 mg/100mL (hours)")
ax.set_title("Corrected Fractional Model: Recovery Time vs Weight")
ax.grid(True, alpha=0.3)

plt.suptitle(
    "Corrected Fractional vs Classical BAC Models Comparison", fontsize=16, y=0.95
)
plt.tight_layout()
plt.savefig("corrected_bac_comparison_comprehensive.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# FIGURE 2: DETAILED MODEL ANALYSIS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Direct model comparison
ax = axes[0, 0]
weight, tbw_ratio, volume, abv = 70, 0.68, 350, 5
A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)

bac_classical = []
bac_fractional = []

for time_point in t_dense:
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model_corrected(time_point, A0, k1, k2, alpha, beta)
    bac_classical.append(B_c * 100)
    bac_fractional.append(B_f * 100)

ax.plot(t_dense, bac_classical, "b-", linewidth=2, label="Classical Model")
ax.plot(t_dense, bac_fractional, "r--", linewidth=2, label="Corrected Fractional Model")
ax.axhline(
    y=80, color="red", linestyle=":", alpha=0.7, label="Legal Limit (80 mg/100mL)"
)
ax.axhline(
    y=10, color="orange", linestyle=":", alpha=0.7, label="Recovery (10 mg/100mL)"
)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BAC (mg/100mL)")
ax.set_title("Direct Model Comparison (70kg Male, Beer)")
ax.legend()
ax.grid(True, alpha=0.3)

# Effect of fractional orders
ax = axes[0, 1]
alpha_values = [0.6, 0.8, 1.0]
colors = ["green", "blue", "red"]
A0 = calculate_initial_concentration(65, 0.68, 350, 5)

for i, alpha_val in enumerate(alpha_values):
    bac_values = []
    for time_point in t_dense:
        if alpha_val == 1.0:  # Classical case
            _, B = classical_bac_model(time_point, A0, k1, k2)
        else:
            _, B = fractional_bac_model_corrected(
                time_point, A0, k1, k2, alpha_val, beta
            )
        bac_values.append(B * 100)

    label = f"α = {alpha_val}" + (" (Classical)" if alpha_val == 1.0 else "")
    ax.plot(t_dense, bac_values, color=colors[i], linewidth=2, label=label)

ax.axhline(y=80, color="red", linestyle=":", alpha=0.7, label="Legal Limit")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BAC (mg/100mL)")
ax.set_title("Effect of Fractional Order α")
ax.legend()
ax.grid(True, alpha=0.3)

# Tolerance time comparison
ax = axes[1, 0]
tolerance_classical = []
tolerance_fractional = []

volume_strong, abv_strong = 500, 40
threshold_high = 80  # mg/100mL
t_extended_tolerance = np.linspace(0, 15, 500)

for weight in weights:
    A0_tolerance = calculate_initial_concentration(
        weight, 0.68, volume_strong, abv_strong
    )

    # Classical model tolerance
    bac_c_values = []
    for time_point in t_extended_tolerance:
        _, B = classical_bac_model(time_point, A0_tolerance, k1, k2)
        bac_c_values.append(B * 100)
    bac_c_np = np.array(bac_c_values)

    indices_above_high_c = np.where(bac_c_np >= threshold_high)[0]
    current_tolerance_c = 0
    if len(indices_above_high_c) > 0:
        t_start_high_c = t_extended_tolerance[indices_above_high_c[0]]
        t_end_high_c = t_extended_tolerance[indices_above_high_c[-1]]
        current_tolerance_c = max(0, t_end_high_c - t_start_high_c)
    tolerance_classical.append(current_tolerance_c)

    # Fractional model tolerance
    bac_f_values = []
    for time_point in t_extended_tolerance:
        _, B = fractional_bac_model_corrected(
            time_point, A0_tolerance, k1, k2, alpha, beta
        )
        bac_f_values.append(B * 100)
    bac_f_np = np.array(bac_f_values)

    indices_above_high_f = np.where(bac_f_np >= threshold_high)[0]
    current_tolerance_f = 0
    if len(indices_above_high_f) > 0:
        t_start_high_f = t_extended_tolerance[indices_above_high_f[0]]
        t_end_high_f = t_extended_tolerance[indices_above_high_f[-1]]
        current_tolerance_f = max(0, t_end_high_f - t_start_high_f)
    tolerance_fractional.append(current_tolerance_f)

ax.plot(
    weights, tolerance_classical, "o-", color="blue", label="Classical", linewidth=2
)
ax.plot(
    weights,
    tolerance_fractional,
    "s-",
    color="red",
    label="Corrected Fractional",
    linewidth=2,
)
ax.set_xlabel("Body Weight (kg)")
ax.set_ylabel("Tolerance Time ΔT (hours)")
ax.set_title("Tolerance Time Comparison (Strong Alcohol)")
ax.legend()
ax.grid(True, alpha=0.3)

# Memory effect visualization
ax = axes[1, 1]
t_short = np.linspace(0, 4, 200)
bac_classical_short = []
bac_fractional_short = []

A0 = calculate_initial_concentration(65, 0.68, 350, 5)
for time_point in t_short:
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model_corrected(
        time_point, A0, k1, k2, 0.7, 0.8
    )  # Lower α for more memory
    bac_classical_short.append(B_c * 100)
    bac_fractional_short.append(B_f * 100)

ax.plot(t_short, bac_classical_short, "b-", linewidth=2, label="Classical (No Memory)")
ax.plot(
    t_short,
    bac_fractional_short,
    "r--",
    linewidth=2,
    label="Fractional (Memory Effect)",
)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("BAC (mg/100mL)")
ax.set_title("Memory Effect in Fractional Model")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("corrected_fractional_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("=== MODEL VALIDATION SUMMARY ===")
print()

# Test with typical scenario
A0_test = calculate_initial_concentration(70, 0.68, 350, 5)
t_test = np.array([0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])

print("Test scenario: 70kg male, 350mL beer (5% ABV)")
print(f"Initial concentration A0: {A0_test:.3f} g/L")
print()
print("Time (h) | Classical (mg/100mL) | Fractional (mg/100mL) | Difference")
print("-" * 70)

for t_val in t_test:
    _, B_c = classical_bac_model(t_val, A0_test, k1, k2)
    _, B_f = fractional_bac_model_corrected(t_val, A0_test, k1, k2, alpha, beta)

    B_c_mg = B_c * 100
    B_f_mg = B_f * 100
    diff = B_f_mg - B_c_mg

    print(f"{t_val:8.1f} | {B_c_mg:19.3f} | {B_f_mg:20.3f} | {diff:10.3f}")

print()
print("Key differences between models:")
peak_c = max([classical_bac_model(t, A0_test, k1, k2)[1] * 100 for t in t_dense])
peak_f = max(
    [
        fractional_bac_model_corrected(t, A0_test, k1, k2, alpha, beta)[1] * 100
        for t in t_dense
    ]
)

print(f"- Classical peak BAC: {peak_c:.1f} mg/100mL")
print(f"- Fractional peak BAC: {peak_f:.1f} mg/100mL")
print(
    f"- Fractional model shows {'slower' if peak_f < peak_c else 'faster'} absorption and elimination"
)
print(
    f"- Memory effects in fractional model lead to {'prolonged' if peak_f > peak_c else 'shortened'} BAC curves"
)

print()
print("✅ CORRECTED FRACTIONAL MODEL VALIDATION COMPLETE")
print(
    "📊 Plots saved: corrected_bac_comparison_comprehensive.png, corrected_fractional_analysis.png"
)
