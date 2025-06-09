import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set plot parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 12)
sns.set_style("whitegrid")

# Mittag-Leffler function approximation
def mittag_leffler(z, alpha, n_terms=30):
    """Single-parameter Mittag-Leffler function E_α(z)"""
    if abs(z) > 10:  # Prevent overflow for large z
        return 0
    result = 0
    for n in range(n_terms):
        try:
            term = (z**n) / gamma(alpha*n + 1)
            if abs(term) < 1e-15:  # Convergence check
                break
            result += term
        except (OverflowError, RuntimeError):
            break
    return result

def two_param_mittag_leffler(x, y, alpha, beta, n_terms=15):
    """Two-parameter Mittag-Leffler function E^(2)_α,β(x,y)"""
    if abs(x) > 5 or abs(y) > 5:  # Prevent overflow
        return 0
    result = 0
    for m in range(n_terms):
        for n in range(n_terms):
            try:
                term = (x**m * y**n) / gamma(alpha*m + beta*n + 1)
                if abs(term) < 1e-15:
                    continue
                result += term
            except (OverflowError, RuntimeError):
                continue
    return result

def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration A0 in stomach"""
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv/100) * rho_ethanol  # grams
    tbw_volume = tbw_ratio * weight  # liters
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0

def classical_bac_model(t, A0, k1, k2):
    """Classical first-order kinetic BAC model"""
    if t == 0:
        return A0, 0
    
    A_t = A0 * np.exp(-k1 * t)
    
    if abs(k1 - k2) < 1e-10:
        B_t = k1 * A0 * t * np.exp(-k1 * t) * 0.1  # Convert to mg/100mL
    else:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)) * 0.1
    
    return A_t, max(0, B_t)

def fractional_bac_model(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """Fractional order BAC model using Caputo derivatives"""
    if t == 0:
        return A0, 0
    
    # A(t) = A0 * E_α(-k1 * t^α)
    A_t = A0 * mittag_leffler(-k1 * (t**alpha), alpha)
    
    # B(t) = k1 * A0 * t^(β-1) * E^(2)_α,β(-k1*t^α, -k2*t^β)
    if t > 0:
        time_factor = t**(beta - 1)
        ml_factor = two_param_mittag_leffler(-k1 * (t**alpha), -k2 * (t**beta), alpha, beta)
        B_t = k1 * A0 * time_factor * ml_factor * 0.1  # Convert to mg/100mL
    else:
        B_t = 0
    
    return max(0, A_t), max(0, B_t)

def find_threshold_times(t_array, bac_array, threshold_high=0.08, threshold_low=0.01):
    """Find times when BAC crosses thresholds"""
    t_i = None  # Time when BAC exceeds threshold_high
    t_f = None  # Time when BAC drops below threshold_low
    
    # Find first crossing above high threshold
    over_high = np.where(bac_array >= threshold_high)[0]
    if len(over_high) > 0:
        t_i = t_array[over_high[0]]
    
    # Find last time above low threshold
    above_low = np.where(bac_array > threshold_low)[0]
    if len(above_low) > 0:
        t_f = t_array[above_low[-1]]
    
    return t_i, t_f

############################

# from scipy.special import gamma

# def ml1(z, alpha, terms=50):
#     """Eₐ(z) 단일파라미터 ML"""
#     s = 0
#     for n in range(terms):
#         s += z**n / gamma(alpha*n+1)
#     return s

# def ml2(z, alpha, β, terms=30):
#     """Eₐ,β(z) 이중파라미터 ML"""
#     s = 0
#     for n in range(terms):
#         s += z**n / gamma(alpha*n + β)
#     return s

# def fractional_bac_model(t, A0, k1, k2, alpha=0.8, β=0.9):
#     # A(t)
#     A_t = A0 * ml1(-k1 * t**alpha, alpha)
#     # B(t)
#     if abs(k2 - k1) < 1e-8:
#         B_t = (k1 * A0 / k2) * t**alpha * ml2(-k1*t**alpha, alpha, alpha+1)
#     else:
#         term1 = t**alpha * ml2(-k1*t**alpha, alpha, alpha+1)
#         term2 = t**β * ml2(-k2*t**β, β, β+1)
#         B_t = (k1 * A0 / (k2 - k1)) * (term1 - term2)
#     # mg/100mL 단위로 변환
#     return A_t, np.maximum(0, B_t*0.1)


# Model parameters
k1, k2 = 1.0, 0.12  # Absorption and elimination rates
alpha, beta = 0.8, 0.9  # Fractional orders
t_max = 10  # hours
t = np.linspace(0, t_max, 300)

# =============================================================================
# EXPERIMENT 1: CLASSICAL BAC MODEL
# =============================================================================

print("="*60)
print("EXPERIMENT 1: CLASSICAL BAC MODEL")
print("="*60)

# Exp 1.1: BAC decrease vs time (4 scenarios, 65kg weight)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

scenarios = [
    {"gender": "Male", "abv": 5, "volume": 350, "tbw_ratio": 0.68, "color": "blue", "linestyle": "-"},
    {"gender": "Female", "abv": 5, "volume": 350, "tbw_ratio": 0.55, "color": "red", "linestyle": "-"},
    {"gender": "Male", "abv": 40, "volume": 50, "tbw_ratio": 0.68, "color": "darkblue", "linestyle": "--"},
    {"gender": "Female", "abv": 40, "volume": 50, "tbw_ratio": 0.55, "color": "darkred", "linestyle": "--"}
]

# Plot 1.1: BAC vs Time (Classical)
ax = axes[0, 0]
for scenario in scenarios:
    weight = 65  # kg
    A0 = calculate_initial_concentration(weight, scenario["tbw_ratio"], scenario["volume"], scenario["abv"])
    
    bac_values = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_values.append(B)
    
    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(t, bac_values, color=scenario["color"], linestyle=scenario["linestyle"], 
           linewidth=2, label=label)

ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='Legal Limit (0.08%)')
ax.axhline(y=0.01, color='orange', linestyle=':', alpha=0.7, label='Recovery (0.01%)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Classical Model: BAC vs Time (65kg)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)

# Plot 1.2: Time to BAC < 0.01% vs TBW (Classical)
ax = axes[0, 1]
tbw_values = np.linspace(30, 70, 20)  # Total Body Water in liters
recovery_times_male = []
recovery_times_female = []

for tbw in tbw_values:
    # Male scenario (beer)
    weight_male = tbw / 0.68  # Reverse calculate weight
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)
    
    bac_male = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0_male, k1, k2)
        bac_male.append(B)
    
    _, t_f_male = find_threshold_times(t, np.array(bac_male))
    recovery_times_male.append(t_f_male if t_f_male else 0)
    
    # Female scenario (beer)
    weight_female = tbw / 0.55
    A0_female = calculate_initial_concentration(weight_female, 0.55, 350, 5)
    
    bac_female = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0_female, k1, k2)
        bac_female.append(B)
    
    _, t_f_female = find_threshold_times(t, np.array(bac_female))
    recovery_times_female.append(t_f_female if t_f_female else 0)

ax.plot(tbw_values, recovery_times_male, 'o-', color='blue', label='Male', linewidth=2)
ax.plot(tbw_values, recovery_times_female, 's-', color='red', label='Female', linewidth=2)
ax.set_xlabel('Total Body Water (L)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Classical Model: Recovery Time vs TBW')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 1.3: Time to BAC < 0.01% vs Weight (Classical)
ax = axes[0, 2]
weights = np.arange(60, 101, 5)
recovery_times_weight = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)  # Male, beer
    
    bac_values = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_values.append(B)
    
    _, t_f = find_threshold_times(t, np.array(bac_values))
    recovery_times_weight.append(t_f if t_f else 0)

ax.plot(weights, recovery_times_weight, 'o-', color='green', linewidth=2, markersize=6)
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Classical Model: Recovery Time vs Weight')
ax.grid(True, alpha=0.3)

# =============================================================================
# EXPERIMENT 2: FRACTIONAL BAC MODEL
# =============================================================================

print("\nEXPERIMENT 2: FRACTIONAL BAC MODEL")
print("="*60)

# Plot 2.1: BAC vs Time (Fractional)
ax = axes[1, 0]
for scenario in scenarios:
    weight = 65  # kg
    A0 = calculate_initial_concentration(weight, scenario["tbw_ratio"], scenario["volume"], scenario["abv"])
    
    bac_values = []
    for time_point in t:
        _, B = fractional_bac_model(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B)
    
    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(t, bac_values, color=scenario["color"], linestyle=scenario["linestyle"], 
           linewidth=2, label=label)

ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='Legal Limit (0.08%)')
ax.axhline(y=0.01, color='orange', linestyle=':', alpha=0.7, label='Recovery (0.01%)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Fractional Model: BAC vs Time (65kg)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)

# Plot 2.2: Time to BAC < 0.01% vs TBW (Fractional)
ax = axes[1, 1]
recovery_times_male_frac = []
recovery_times_female_frac = []

for tbw in tbw_values:
    # Male scenario (beer)
    weight_male = tbw / 0.68
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)
    
    bac_male = []
    for time_point in t:
        _, B = fractional_bac_model(time_point, A0_male, k1, k2, alpha, beta)
        bac_male.append(B)
    
    _, t_f_male = find_threshold_times(t, np.array(bac_male))
    recovery_times_male_frac.append(t_f_male if t_f_male else 0)
    
    # Female scenario (beer)
    weight_female = tbw / 0.55
    A0_female = calculate_initial_concentration(weight_female, 0.55, 350, 5)
    
    bac_female = []
    for time_point in t:
        _, B = fractional_bac_model(time_point, A0_female, k1, k2, alpha, beta)
        bac_female.append(B)
    
    _, t_f_female = find_threshold_times(t, np.array(bac_female))
    recovery_times_female_frac.append(t_f_female if t_f_female else 0)

ax.plot(tbw_values, recovery_times_male_frac, 'o-', color='blue', label='Male', linewidth=2)
ax.plot(tbw_values, recovery_times_female_frac, 's-', color='red', label='Female', linewidth=2)
ax.set_xlabel('Total Body Water (L)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Fractional Model: Recovery Time vs TBW')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2.3: Time to BAC < 0.01% vs Weight (Fractional)
ax = axes[1, 2]
recovery_times_weight_frac = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)  # Male, beer
    
    bac_values = []
    for time_point in t:
        _, B = fractional_bac_model(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B)
    
    _, t_f = find_threshold_times(t, np.array(bac_values))
    recovery_times_weight_frac.append(t_f if t_f else 0)

ax.plot(weights, recovery_times_weight_frac, 'o-', color='green', linewidth=2, markersize=6)
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Fractional Model: Recovery Time vs Weight')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# MODEL COMPARISON AND ANALYSIS
# =============================================================================

print("\nMODEL COMPARISON ANALYSIS")
print("="*60)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Comparison 1: Direct model comparison for a typical scenario
ax = axes[0, 0]
weight, tbw_ratio, volume, abv = 70, 0.68, 350, 5  # Typical male drinking beer
A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)

bac_classical = []
bac_fractional = []

for time_point in t:
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model(time_point, A0, k1, k2, alpha, beta)
    bac_classical.append(B_c)
    bac_fractional.append(B_f)

ax.plot(t, bac_classical, 'b-', linewidth=2, label='Classical Model')
ax.plot(t, bac_fractional, 'r--', linewidth=2, label='Fractional Model')
ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7)
ax.axhline(y=0.01, color='orange', linestyle=':', alpha=0.7)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Model Comparison: 70kg Male, Beer')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparison 2: Tolerance time comparison
ax = axes[0, 1]
tolerance_classical = []
tolerance_fractional = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    
    # Classical model tolerance
    bac_c = []
    for time_point in t:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_c.append(B)
    t_i_c, t_f_c = find_threshold_times(t, np.array(bac_c))
    tolerance_c = (t_f_c - t_i_c) if (t_i_c and t_f_c) else 0
    tolerance_classical.append(tolerance_c)
    
    # Fractional model tolerance
    bac_f = []
    for time_point in t:
        _, B = fractional_bac_model(time_point, A0, k1, k2, alpha, beta)
        bac_f.append(B)
    t_i_f, t_f_f = find_threshold_times(t, np.array(bac_f))
    tolerance_f = (t_f_f - t_i_f) if (t_i_f and t_f_f) else 0
    tolerance_fractional.append(tolerance_f)

ax.plot(weights, tolerance_classical, 'o-', color='blue', label='Classical', linewidth=2)
ax.plot(weights, tolerance_fractional, 's-', color='red', label='Fractional', linewidth=2)
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Tolerance Time ΔT (hours)')
ax.set_title('Tolerance Time Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Effect of fractional orders
ax = axes[1, 0]
alpha_values = [0.6, 0.8, 1.0]  # alpha = 1.0 reduces to classical
colors = ['green', 'blue', 'red']
A0 = calculate_initial_concentration(70, 0.68, 350, 5)

for i, alpha_val in enumerate(alpha_values):
    bac_values = []
    for time_point in t:
        if alpha_val == 1.0:
            _, B = classical_bac_model(time_point, A0, k1, k2)
        else:
            _, B = fractional_bac_model(time_point, A0, k1, k2, alpha_val, beta)
        bac_values.append(B)
    
    label = f'α = {alpha_val}' + (' (Classical)' if alpha_val == 1.0 else '')
    ax.plot(t, bac_values, color=colors[i], linewidth=2, label=label)

ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Effect of Fractional Order α')
ax.legend()
ax.grid(True, alpha=0.3)

# Memory effect visualization
ax = axes[1, 1]
# Show how fractional model captures memory effects through slower dynamics
t_short = np.linspace(0, 4, 100)
bac_classical_short = []
bac_fractional_short = []

A0 = calculate_initial_concentration(70, 0.68, 350, 5)
for time_point in t_short:
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model(time_point, A0, k1, k2, 0.7, 0.8)  # Lower orders for more memory
    bac_classical_short.append(B_c)
    bac_fractional_short.append(B_f)

ax.plot(t_short, bac_classical_short, 'b-', linewidth=2, label='Classical (No Memory)')
ax.plot(t_short, bac_fractional_short, 'r--', linewidth=2, label='Fractional (Memory Effect)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Memory Effect in Fractional Model')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\nSUMMARY STATISTICS")
print("="*60)

# Example calculations for a standard scenario
weight, tbw_ratio, volume, abv = 70, 0.68, 350, 5
A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)

print(f"Standard Scenario: {weight}kg male drinking {volume}mL beer ({abv}% ABV)")
print(f"Initial stomach concentration (A₀): {A0:.4f} g/L")
print(f"Model parameters: k₁={k1}, k₂={k2}, α={alpha}, β={beta}")
print()

# Classical model results
bac_c = []
for time_point in t:
    _, B = classical_bac_model(time_point, A0, k1, k2)
    bac_c.append(B)
bac_c = np.array(bac_c)

peak_idx_c = np.argmax(bac_c)
peak_time_c = t[peak_idx_c]
peak_bac_c = bac_c[peak_idx_c]

t_i_c, t_f_c = find_threshold_times(t, bac_c)
tolerance_c = (t_f_c - t_i_c) if (t_i_c and t_f_c) else 0

print("Classical Model Results:")
print(f"  Peak BAC: {peak_bac_c:.3f} mg/100mL at {peak_time_c:.1f} hours")
if t_i_c: print(f"  Time to 0.08 mg/100mL: {t_i_c:.1f} hours")
if t_f_c: print(f"  Time to recover (< 0.01 mg/100mL): {t_f_c:.1f} hours")
print(f"  Tolerance time (ΔT): {tolerance_c:.1f} hours")
print()

# Fractional model results
bac_f = []
for time_point in t:
    _, B = fractional_bac_model(time_point, A0, k1, k2, alpha, beta)
    bac_f.append(B)
bac_f = np.array(bac_f)

peak_idx_f = np.argmax(bac_f)
peak_time_f = t[peak_idx_f]
peak_bac_f = bac_f[peak_idx_f]

t_i_f, t_f_f = find_threshold_times(t, bac_f)
tolerance_f = (t_f_f - t_i_f) if (t_i_f and t_f_f) else 0

print("Fractional Model Results:")
print(f"  Peak BAC: {peak_bac_f:.3f} mg/100mL at {peak_time_f:.1f} hours")
if t_i_f: print(f"  Time to 0.08 mg/100mL: {t_i_f:.1f} hours")
if t_f_f: print(f"  Time to recover (< 0.01 mg/100mL): {t_f_f:.1f} hours")
print(f"  Tolerance time (ΔT): {tolerance_f:.1f} hours")
print()

print("Key Differences:")
print(f"  Peak BAC difference: {abs(peak_bac_f - peak_bac_c):.3f} mg/100mL")
print(f"  Peak time difference: {abs(peak_time_f - peak_time_c):.1f} hours")
print(f"  Tolerance time difference: {abs(tolerance_f - tolerance_c):.1f} hours")
print()

print("Physiological Interpretation:")
print("- The fractional model captures non-local memory effects in alcohol metabolism")
print("- Lower fractional orders (α, β < 1) represent slower, more realistic dynamics")
print("- Memory effects lead to prolonged BAC curves compared to classical exponential decay")
print("- This provides more accurate predictions for individual alcohol tolerance")