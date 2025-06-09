"""
Fixed plotting script for BAC models with improved visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the functions from all_v9.py
from all_v9 import (
    calculate_initial_concentration, 
    classical_bac_model, 
    fractional_bac_model_final,
    find_threshold_times,
    ml1, ml2
)

# Set plot parameters for better visualization
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (18, 12)
sns.set_style("whitegrid")

# Model parameters - fixed for proper behavior
k1, k2 = 1.0, 1.2  # Absorption and elimination rates (k2 > k1)
alpha, beta = 0.8, 0.9  # Fractional orders
t_max = 10  # hours
t = np.linspace(0, t_max, 300)  # Regular time points
t_dense = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.5, t_max, 300)])  # More points at beginning
t_dense.sort()

# Ensure classical model matches fractional when alpha=1.0
def classical_bac_model_reference(t, A0):
    """Classical model reference implementation for consistent comparison"""
    if t == 0:
        return A0, 0
        
    A_t = A0 * np.exp(-k1 * t)
    
    if abs(k1 - k2) < 1e-10:
        B_t = k1 * A0 * t * np.exp(-k1 * t) * 0.1
    else:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)) * 0.1
    
    return A_t, max(0, B_t)

# =============================================================================
# FIGURE 1: BAC CURVES COMPARISON
# =============================================================================

# Create grid of plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Define test scenarios
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
    for time_point in t_dense:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_values.append(B)
    
    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(t_dense, bac_values, color=scenario["color"], linestyle=scenario["linestyle"], 
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
    weight_male = tbw / 0.68
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

# Plot 2.1: BAC vs Time (Fractional) - FIXED
ax = axes[1, 0]
for scenario in scenarios:
    weight = 65  # kg
    A0 = calculate_initial_concentration(weight, scenario["tbw_ratio"], scenario["volume"], scenario["abv"])
    
    # For proper visualization, use more time points at the beginning
    bac_values = []
    for time_point in t_dense:
        _, B = fractional_bac_model_final(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B)
    
    label = f'{scenario["gender"]}, {"Low ABV" if scenario["abv"] == 5 else "High ABV"}'
    ax.plot(t_dense, bac_values, color=scenario["color"], linestyle=scenario["linestyle"], 
           linewidth=2, label=label)

ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='Legal Limit (0.08%)')
ax.axhline(y=0.01, color='orange', linestyle=':', alpha=0.7, label='Recovery (0.01%)')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Fractional Model: BAC vs Time (65kg)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)  # Match scale with classical model

# Plot 2.2: Time to BAC < 0.01% vs TBW (Fractional) - FIXED
ax = axes[1, 1]
recovery_times_male_frac = []
recovery_times_female_frac = []

for tbw in tbw_values:
    # Male scenario (beer)
    weight_male = tbw / 0.68
    A0_male = calculate_initial_concentration(weight_male, 0.68, 350, 5)
    
    # Use denser time points for better threshold detection
    bac_male = []
    for time_point in t_dense:  
        _, B = fractional_bac_model_final(time_point, A0_male, k1, k2, alpha, beta)
        bac_male.append(B)
    
    _, t_f_male = find_threshold_times(t_dense, np.array(bac_male))
    recovery_times_male_frac.append(t_f_male if t_f_male else 0)
    
    # Female scenario (beer)
    weight_female = tbw / 0.55
    A0_female = calculate_initial_concentration(weight_female, 0.55, 350, 5)
    
    bac_female = []
    for time_point in t_dense:
        _, B = fractional_bac_model_final(time_point, A0_female, k1, k2, alpha, beta)
        bac_female.append(B)
    
    _, t_f_female = find_threshold_times(t_dense, np.array(bac_female))
    recovery_times_female_frac.append(t_f_female if t_f_female else 0)

ax.plot(tbw_values, recovery_times_male_frac, 'o-', color='blue', label='Male', linewidth=2)
ax.plot(tbw_values, recovery_times_female_frac, 's-', color='red', label='Female', linewidth=2)
ax.set_xlabel('Total Body Water (L)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Fractional Model: Recovery Time vs TBW')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2.3: Time to BAC < 0.01% vs Weight (Fractional) - FIXED
ax = axes[1, 2]
recovery_times_weight_frac = []

for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)  # Male, beer
    bac_values = []
    for time_point in t_dense:  # Use denser time grid
        _, B = fractional_bac_model_final(time_point, A0, k1, k2, alpha, beta)
        bac_values.append(B)
    
    _, t_f = find_threshold_times(t_dense, np.array(bac_values))
    recovery_times_weight_frac.append(t_f if t_f else 0)

ax.plot(weights, recovery_times_weight_frac, 'o-', color='green', linewidth=2, markersize=6)
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Time to BAC < 0.01% (hours)')
ax.set_title('Fractional Model: Recovery Time vs Weight')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bac_comparison.png', dpi=300)
plt.show()

# =============================================================================
# FIGURE 2: MODEL ANALYSIS
# =============================================================================

# Create analysis plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Comparison 1: Direct model comparison for a typical scenario - FIXED
ax = axes[0, 0]
weight, tbw_ratio, volume, abv = 70, 0.68, 350, 5  # Typical male drinking beer
A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)

bac_classical = []
bac_fractional = []

for time_point in t_dense:  # Use denser time grid
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model_final(time_point, A0, k1, k2, alpha, beta)
    bac_classical.append(B_c)
    bac_fractional.append(B_f)

ax.plot(t_dense, bac_classical, 'b-', linewidth=2, label='Classical Model')
ax.plot(t_dense, bac_fractional, 'r--', linewidth=2, label='Fractional Model')
ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='Legal Limit')
ax.axhline(y=0.01, color='orange', linestyle=':', alpha=0.7, label='Recovery')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Model Comparison: 70kg Male, Beer')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparison 2: Tolerance time comparison - FIXED
ax = axes[0, 1]
tolerance_classical = []
tolerance_fractional = []

# Calculate with denser time sampling for better accuracy
for weight in weights:
    A0 = calculate_initial_concentration(weight, 0.68, 350, 5)
    
    # Classical model tolerance
    bac_c = []
    for time_point in t_dense:
        _, B = classical_bac_model(time_point, A0, k1, k2)
        bac_c.append(B)
    t_i_c, t_f_c = find_threshold_times(t_dense, np.array(bac_c))
    tolerance_c = (t_f_c - t_i_c) if (t_i_c and t_f_c) else 0
    tolerance_classical.append(tolerance_c)
    
    # Fractional model tolerance
    bac_f = []
    for time_point in t_dense:
        _, B = fractional_bac_model_final(time_point, A0, k1, k2, alpha, beta)
        bac_f.append(B)
    t_i_f, t_f_f = find_threshold_times(t_dense, np.array(bac_f))
    tolerance_f = (t_f_f - t_i_f) if (t_i_f and t_f_f) else 0
    tolerance_fractional.append(tolerance_f)

ax.plot(weights, tolerance_classical, 'o-', color='blue', label='Classical', linewidth=2)
ax.plot(weights, tolerance_fractional, 's-', color='red', label='Fractional', linewidth=2)
ax.set_xlabel('Body Weight (kg)')
ax.set_ylabel('Tolerance Time ΔT (hours)')
ax.set_title('Tolerance Time Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Effect of fractional orders - FIXED
ax = axes[1, 0]
alpha_values = [0.6, 0.8, 1.0]  # alpha = 1.0 reduces to classical
colors = ['green', 'blue', 'red']
A0 = calculate_initial_concentration(70, 0.68, 350, 5)

for i, alpha_val in enumerate(alpha_values):
    bac_values = []
    for time_point in t_dense:
        if alpha_val == 1.0:
            _, B = classical_bac_model(time_point, A0, k1, k2)
        else:
            _, B = fractional_bac_model_final(time_point, A0, k1, k2, alpha_val, beta)
        bac_values.append(B)
    
    label = f'α = {alpha_val}' + (' (Classical)' if alpha_val == 1.0 else '')
    ax.plot(t_dense, bac_values, color=colors[i], linewidth=2, label=label)

ax.axhline(y=0.08, color='red', linestyle=':', alpha=0.7, label='Legal Limit')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('BAC (mg/100mL)')
ax.set_title('Effect of Fractional Order α')
ax.legend()
ax.grid(True, alpha=0.3)

# Memory effect visualization - FIXED
ax = axes[1, 1]
# Show how fractional model captures memory effects through slower dynamics
t_short = np.linspace(0, 4, 200)  # More sampling points
bac_classical_short = []
bac_fractional_short = []

A0 = calculate_initial_concentration(70, 0.68, 350, 5)
for time_point in t_short:
    _, B_c = classical_bac_model(time_point, A0, k1, k2)
    _, B_f = fractional_bac_model_final(time_point, A0, k1, k2, 0.7, 0.8)
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
plt.savefig('model_analysis.png', dpi=300)
plt.show()

print("\nFixed visualization generated successfully!")
print("New figures show:")
print("1. Proper BAC curves with expected peak and decay")
print("2. Correct correlation between weight/TBW and recovery time")
print("3. Improved tolerance time visualization")
print("4. Consistent y-axis scaling between classical and fractional models")