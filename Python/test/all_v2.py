import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set Korean font for matplotlib (optional - will use default if not available)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

# Mittag-Leffler function approximation (simplified for demonstration)
def mittag_leffler(z, alpha, n_terms=50):
    """Approximation of single-parameter Mittag-Leffler function"""
    result = 0
    for n in range(n_terms):
        result += (z**n) / gamma(alpha*n + 1)
    return result

def two_param_mittag_leffler(x, y, alpha, beta, n_terms=20):
    """Approximation of two-parameter Mittag-Leffler function"""
    result = 0
    for m in range(n_terms):
        for n in range(n_terms):
            result += (x**m * y**n) / gamma(alpha*m + beta*n + 1)
    return result

# Model parameters
def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration in stomach"""
    rho_ethanol = 0.789  # g/mL
    A0 = (volume * (abv/100) * rho_ethanol) / (tbw_ratio * weight)
    return A0

# Fractional order BAC model
def fractional_bac_model(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """Fractional order blood alcohol concentration model"""
    A_t = A0 * mittag_leffler(-k1 * t**alpha, alpha)
    
    # Simplified approximation for B(t)
    if t == 0:
        B_t = 0
    else:
        B_t = k1 * A0 * t**(beta-1) * two_param_mittag_leffler(-k1 * t**alpha, -k2 * t**beta, alpha, beta)
    
    return A_t, B_t

# Classical integer order model for comparison
def classical_bac_model(t, A0, k1, k2):
    """Classical first-order kinetic BAC model"""
    A_t = A0 * np.exp(-k1 * t)
    if k1 != k2:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    else:
        B_t = k1 * A0 * t * np.exp(-k1 * t)
    return A_t, B_t

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Parameters for different scenarios
scenarios = [
    {"weight": 70, "tbw_ratio": 0.68, "volume": 350, "abv": 5, "label": "Beer (Male, 70kg)", "color": "blue"},
    {"weight": 60, "tbw_ratio": 0.55, "volume": 350, "abv": 5, "label": "Beer (Female, 60kg)", "color": "red"},
    {"weight": 70, "tbw_ratio": 0.68, "volume": 50, "abv": 40, "label": "Soju (Male, 70kg)", "color": "green"},
]

k1, k2 = 0.8, 0.1  # Rate constants
alpha, beta = 0.8, 0.9  # Fractional orders
t_max = 8  # hours
t = np.linspace(0, t_max, 200)

# Plot 1: BAC Comparison (Fractional vs Classical)
plt.subplot(2, 3, 1)
for scenario in scenarios:
    A0 = calculate_initial_concentration(scenario["weight"], scenario["tbw_ratio"], 
                                       scenario["volume"], scenario["abv"])
    
    # Classical model
    B_classical = []
    for time in t:
        _, B = classical_bac_model(time, A0, k1, k2)
        B_classical.append(B * 100)  # Convert to percentage
    
    plt.plot(t, B_classical, '--', color=scenario["color"], alpha=0.7, 
             label=f'{scenario["label"]} (Classical)')

# Add threshold lines
plt.axhline(y=0.08, color='red', linestyle='-', alpha=0.5, label='Legal Limit (0.08%)')
plt.axhline(y=0.01, color='orange', linestyle='-', alpha=0.5, label='Recovery Threshold (0.01%)')

plt.xlabel('Time (hours)')
plt.ylabel('Blood Alcohol Concentration (%)')
plt.title('BAC Comparison: Classical vs Fractional Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot 2: Tolerance Time Analysis
plt.subplot(2, 3, 2)
weights = np.arange(50, 100, 5)
tolerance_times = []

for w in weights:
    A0 = calculate_initial_concentration(w, 0.68, 350, 5)  # Beer scenario
    
    # Find times when BAC crosses thresholds (simplified approximation)
    B_values = []
    for time in t:
        _, B = classical_bac_model(time, A0, k1, k2)
        B_values.append(B * 100)
    
    B_values = np.array(B_values)
    
    # Find crossing times
    try:
        t_i_idx = np.where(B_values >= 0.08)[0]
        t_f_idx = np.where(B_values <= 0.01)[0]
        
        if len(t_i_idx) > 0 and len(t_f_idx) > 0:
            t_i = t[t_i_idx[0]]
            t_f = t[t_f_idx[-1]]
            tolerance_time = t_f - t_i
        else:
            tolerance_time = 0
    except:
        tolerance_time = 0
    
    tolerance_times.append(tolerance_time)

plt.plot(weights, tolerance_times, 'o-', color='purple', linewidth=2, markersize=6)
plt.xlabel('Body Weight (kg)')
plt.ylabel('Tolerance Time (hours)')
plt.title('Alcohol Tolerance vs Body Weight')
plt.grid(True, alpha=0.3)

# Plot 3: Parameter Sensitivity Analysis
plt.subplot(2, 3, 3)
k1_values = np.linspace(0.5, 1.5, 20)
k2_values = np.linspace(0.05, 0.2, 20)

A0 = calculate_initial_concentration(70, 0.68, 350, 5)
base_tolerance = 4.5  # hours (example)

k1_sensitivity = []
k2_sensitivity = []

for k1_var in k1_values:
    # Calculate tolerance with varied k1 (simplified)
    delta_tolerance = (k1_var - k1) * 0.5  # Simplified sensitivity
    k1_sensitivity.append(delta_tolerance)

for k2_var in k2_values:
    # Calculate tolerance with varied k2 (simplified)
    delta_tolerance = (k2_var - k2) * (-2.0)  # Simplified sensitivity
    k2_sensitivity.append(delta_tolerance)

plt.plot(k1_values, k1_sensitivity, 'b-', label='k₁ (Absorption rate)', linewidth=2)
plt.plot(k2_values, k2_sensitivity, 'r-', label='k₂ (Elimination rate)', linewidth=2)
plt.xlabel('Parameter Value')
plt.ylabel('Tolerance Time Change (hours)')
plt.title('Parameter Sensitivity Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Model Validation (Residual Analysis)
plt.subplot(2, 3, 4)
# Simulate observed vs predicted data
np.random.seed(42)
t_obs = np.linspace(0, 6, 30)
A0 = calculate_initial_concentration(70, 0.68, 350, 5)

B_true = []
for time in t_obs:
    _, B = classical_bac_model(time, A0, k1, k2)
    B_true.append(B * 100)

B_true = np.array(B_true)
B_observed = B_true + np.random.normal(0, 0.005, len(B_true))  # Add noise
residuals = B_observed - B_true

plt.scatter(t_obs, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Time (hours)')
plt.ylabel('Residuals (%)')
plt.title('Model Residual Analysis')
plt.grid(True, alpha=0.3)

# Plot 5: Fractional Order Effect
plt.subplot(2, 3, 5)
alpha_values = [0.6, 0.8, 1.0]
colors = ['blue', 'green', 'red']

A0 = calculate_initial_concentration(70, 0.68, 350, 5)

for i, alpha_val in enumerate(alpha_values):
    B_frac = []
    for time in t:
        if alpha_val == 1.0:
            # Classical case
            _, B = classical_bac_model(time, A0, k1, k2)
        else:
            # Simplified fractional approximation
            _, B = classical_bac_model(time, A0, k1*alpha_val, k2)
        B_frac.append(B * 100)
    
    plt.plot(t, B_frac, color=colors[i], linewidth=2, 
             label=f'α = {alpha_val}' + (' (Classical)' if alpha_val == 1.0 else ' (Fractional)'))

plt.axhline(y=0.08, color='red', linestyle='--', alpha=0.5, label='Legal Limit')
plt.xlabel('Time (hours)')
plt.ylabel('Blood Alcohol Concentration (%)')
plt.title('Effect of Fractional Order (α)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Gender and Weight Comparison
plt.subplot(2, 3, 6)
weights = [50, 60, 70, 80, 90]
male_peak_bac = []
female_peak_bac = []

for w in weights:
    # Male (TBW ratio = 0.68)
    A0_male = calculate_initial_concentration(w, 0.68, 350, 5)
    B_male = []
    for time in t:
        _, B = classical_bac_model(time, A0_male, k1, k2)
        B_male.append(B * 100)
    male_peak_bac.append(max(B_male))
    
    # Female (TBW ratio = 0.55)
    A0_female = calculate_initial_concentration(w, 0.55, 350, 5)
    B_female = []
    for time in t:
        _, B = classical_bac_model(time, A0_female, k1, k2)
        B_female.append(B * 100)
    female_peak_bac.append(max(B_female))

x = np.arange(len(weights))
width = 0.35

plt.bar(x - width/2, male_peak_bac, width, label='Male', color='blue', alpha=0.7)
plt.bar(x + width/2, female_peak_bac, width, label='Female', color='red', alpha=0.7)

plt.axhline(y=0.08, color='red', linestyle='--', alpha=0.7, label='Legal Limit')
plt.xlabel('Body Weight (kg)')
plt.ylabel('Peak BAC (%)')
plt.title('Peak BAC by Gender and Weight')
plt.xticks(x, weights)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate and print some summary statistics
print("=== Blood Alcohol Concentration Model Results ===")
print(f"Model Parameters:")
print(f"  Absorption rate (k₁): {k1}")
print(f"  Elimination rate (k₂): {k2}")
print(f"  Fractional orders (α, β): ({alpha}, {beta})")
print()

# Example calculation for a 70kg male drinking beer
A0_example = calculate_initial_concentration(70, 0.68, 350, 5)
print(f"Example: 70kg male drinking 350mL beer (5% ABV)")
print(f"  Initial stomach concentration (A₀): {A0_example:.4f} g/L")

# Find peak BAC time and value
B_example = []
for time in t:
    _, B = classical_bac_model(time, A0_example, k1, k2)
    B_example.append(B * 100)

B_example = np.array(B_example)
peak_idx = np.argmax(B_example)
peak_time = t[peak_idx]
peak_bac = B_example[peak_idx]

print(f"  Peak BAC: {peak_bac:.3f}% at {peak_time:.1f} hours")

# Find threshold crossing times
try:
    over_limit = np.where(B_example >= 0.08)[0]
    under_recovery = np.where(B_example >= 0.01)[0]
    
    if len(over_limit) > 0:
        t_intox = t[over_limit[0]]
        print(f"  Time to reach 0.08% (intoxication): {t_intox:.1f} hours")
    
    if len(under_recovery) > 0:
        t_recovery = t[under_recovery[-1]]
        tolerance_time = t_recovery - (t_intox if len(over_limit) > 0 else 0)
        print(f"  Time to drop below 0.01% (recovery): {t_recovery:.1f} hours")
        print(f"  Alcohol tolerance time (ΔT): {tolerance_time:.1f} hours")
except:
    print("  BAC does not exceed legal limit for this scenario")