import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration and Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

# Set default font (DejaVu Sans supports most Unicode characters)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

def mittag_leffler(z, alpha, n_terms=50):
    """
    Single-parameter Mittag–Leffler function approximation:
      E_{α}(z) = ∑_{n=0}^{∞} z^n / Γ(α · n + 1)
    """
    result = 0.0
    for n in range(n_terms):
        term = (z**n) / gamma(alpha * n + 1)
        result += term
        if abs(term) < 1e-12:
            break
    return result

def two_param_mittag_leffler(x, y, alpha, beta, n_terms=20):
    """
    Two-parameter Mittag–Leffler function approximation:
      E^{(2)}_{α,β}(x, y) = ∑_{m=0}^{∞} ∑_{n=0}^{∞} (x^m · y^n) / Γ(α m + β n + 1)
    """
    total = 0.0
    for m in range(n_terms):
        for n in range(n_terms):
            exponent = alpha * m + beta * n
            term = (x**m * y**n) / gamma(exponent + 1)
            total += term
            if abs(term) < 1e-14:
                break
        if abs((x**m) / gamma(alpha * m + 1)) < 1e-14:
            break
    return total

def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """
    Compute initial alcohol concentration in stomach (units: g/L):
      A0 = (V [mL] × (ABV/100) × ρ_EtOH [g/mL]) / (r [L/kg] × weight [kg])
      where ρ_EtOH = 0.789 g/mL.
    """
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv / 100.0) * rho_ethanol
    tbw_volume = tbw_ratio * weight
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0

def classical_bac_model(t, A0, k1, k2):
    """
    Classical integer-order BAC model:
      A(t) = A0 · e^(−k1 t)
      B(t) = (k1 · A0 / (k2 − k1)) · (e^(−k1 t) − e^(−k2 t))
    Returns:
      A_t [g/L], B_t [g/dL]
      (Note: dividing by 10 converts A0 from g/L to g/dL before computing B.)
    """
    A_t = A0 * np.exp(-k1 * t)
    if abs(k2 - k1) > 1e-12:
        B_t_g_per_dL = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)) / 10.0
    else:
        B_t_g_per_dL = (k1 * A0 * t * np.exp(-k1 * t)) / 10.0
    return A_t, B_t_g_per_dL

def fractional_bac_model(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """
    Caputo-fractional-order BAC model:
      A(t)        = A0 · E_{α}(−k1 t^α)
      B(t) [g/L]  = k1 · A0 · t^(β−1) · E^{(2)}_{α,β}(−k1 t^α, −k2 t^β)
    Returns:
      A_t [g/L], B_t [g/L]
    """
    A_t = A0 * mittag_leffler(-k1 * t**alpha, alpha)
    if t < 1e-15:
        B_t = 0.0
    else:
        x = -k1 * (t**alpha)
        y = -k2 * (t**beta)
        E2 = two_param_mittag_leffler(x, y, alpha, beta)
        B_t = k1 * A0 * (t ** (beta - 1)) * E2
    return A_t, B_t

def find_threshold_time(bac_array, time_grid, threshold):
    """
    Identify the first index where bac_array >= threshold (mg/dL).
    If no crossing is found, return None.
    Otherwise return corresponding time.
    """
    idx = np.where(bac_array >= threshold)[0]
    return time_grid[idx[0]] if idx.size > 0 else None

# ──────────────────────────────────────────────────────────────────────────────
# 2. Simulation Parameters
# ──────────────────────────────────────────────────────────────────────────────

# Scenarios: (weight [kg], TBW ratio, volume [mL], ABV [%], label, color)
scenarios = [
    {"weight": 70, "tbw_ratio": 0.68, "volume": 350, "abv": 5,  "label": "Beer (Male, 70 kg)",    "color": "tab:blue"},
    {"weight": 60, "tbw_ratio": 0.55, "volume": 350, "abv": 5,  "label": "Beer (Female, 60 kg)",  "color": "tab:red"},
    {"weight": 70, "tbw_ratio": 0.68, "volume":  50, "abv": 40, "label": "Soju (Male, 70 kg)",    "color": "tab:green"},
]

k1, k2   = 1.2, 0.15     # Absorption and elimination constants
alpha, beta = 0.8, 0.9   # Fractional orders
T_max    = 8.0           # Simulation duration in hours
J        = 200           # Number of time steps
t_grid   = np.linspace(0, T_max, J)  # Time vector (0 ≤ t ≤ 8 h)

# Thresholds (mg/dL)
BAC_high   = 0.08   # Intoxication limit (0.08 mg/dL = 0.0008 g/dL)
BAC_low    = 0.01   # Recovery limit    (0.01 mg/dL = 0.0001 g/dL)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute and Plot Results
# ──────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))

# --- Subplot 1: Classical vs. Fractional BAC Curves ---
ax1 = plt.subplot(2, 3, 1)
for scenario in scenarios:
    w      = scenario["weight"]
    r      = scenario["tbw_ratio"]
    V      = scenario["volume"]
    abv    = scenario["abv"]
    color  = scenario["color"]
    label  = scenario["label"]

    # 1) Compute A0 [g/L]
    A0 = calculate_initial_concentration(w, r, V, abv)

    # 2) Evaluate classical model → B_classical_mg_dL (mg/dL)
    B_classical = np.zeros_like(t_grid)
    for i, t in enumerate(t_grid):
        _, B_g_per_dL = classical_bac_model(t, A0, k1, k2)
        B_classical[i] = B_g_per_dL * 1000  # Convert g/dL → mg/dL

    # Plot classical (dashed)
    ax1.plot(
        t_grid, B_classical, '--', color=color, alpha=0.7,
        label=f'{label} (Classical)'
    )

    # 3) Evaluate fractional model → B_fractional_mg_dL (mg/dL)
    B_fractional = np.zeros_like(t_grid)
    for i, t in enumerate(t_grid):
        _, B_g_per_L = fractional_bac_model(t, A0, k1, k2, alpha, beta)
        B_fractional[i] = B_g_per_L * 100      # Convert g/L → mg/dL

    # Plot fractional (solid)
    ax1.plot(
        t_grid, B_fractional, color=color, linewidth=2,
        label=f'{label} (Fractional)'
    )

# Draw threshold lines (mg/dL)
ax1.axhline(y=BAC_high, color='red',   linestyle='-', alpha=0.5, label='Legal Limit 0.08 mg/dL')
ax1.axhline(y=BAC_low,  color='orange',linestyle='-', alpha=0.5, label='Recovery 0.01 mg/dL')

ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Blood Alcohol Concentration (mg/100 mL)')
ax1.set_title('BAC Comparison: Classical vs. Fractional Models')
ax1.set_ylim(0, 100)  # Display up to 100 mg/dL = 0.10 g/dL
ax1.legend(loc='upper right', fontsize='small')
ax1.grid(alpha=0.3)

# --- Subplot 2: Tolerance Time vs. Body Weight (Classical Model, 수정본) ---
plt.subplot(2, 3, 2)

weights = np.arange(50, 100, 5)    # 50kg, 55kg, …, 95kg
tolerance_times = []

for w in weights:
    # 1) A0 계산 (beer scenario: 350 mL, 5 % ABV)
    A0 = calculate_initial_concentration(w, 0.68, 350, 5)

    # 2) 시간 그리드에서 classical B(t)를 "mg/dL"로 계산
    B_vals_mg_dL = np.zeros_like(t_grid)
    for i, tau in enumerate(t_grid):
        _, B_g_per_dL = classical_bac_model(tau, A0, k1, k2)
        B_vals_mg_dL[i] = B_g_per_dL * 1000   # g/dL → mg/dL

    # 3) 임계치를 “mg/dL”로 맞춘 후 첫 번째, 마지막 crossing 시점 찾기
    # (a) t_i: B ≥ 0.08 mg/dL 되는 첫 시각
    idx_i = np.where(B_vals_mg_dL >= 0.08)[0]
    t_i   = t_grid[idx_i[0]] if idx_i.size > 0 else None

    # (b) t_f: B ≥ 0.01 mg/dL 상태가 끝나는 마지막 시각
    idx_low = np.where(B_vals_mg_dL >= 0.01)[0]
    t_f     = t_grid[idx_low[-1]] if idx_low.size > 0 else None

    # 4) ΔT 계산: 만약 t_i, t_f 모두 존재하고 t_f > t_i 이면 (t_f − t_i), 
    #    아니면 “B > 0.01” 구간 전체 길이를 ΔT로 설정
    if (t_i is not None) and (t_f is not None) and (t_f > t_i):
        ΔT = t_f - t_i
    elif idx_low.size > 0:
        ΔT = t_grid[idx_low[-1]] - t_grid[idx_low[0]]
    else:
        ΔT = 0.0

    tolerance_times.append(ΔT)

# 5) 완성된 ΔT를 체중별로 플롯
plt.plot(
    weights, tolerance_times, 'o-', color='tab:purple',
    linewidth=2, markersize=6
)
plt.xlabel('Body Weight (kg)')
plt.ylabel('Tolerance Time ΔT (hours)')
plt.title('Alcohol Tolerance vs. Body Weight (Classical Model)')
plt.grid(alpha=0.3)


# --- Subplot 3: Parameter Sensitivity (Simplified Linear Approximation) ---
ax3 = plt.subplot(2, 3, 3)
k1_values = np.linspace(0.5, 1.5, 20)
k2_values = np.linspace(0.05, 0.20, 20)

# Compute sensitivity: ΔΔT/Δθ (simplified approximation)
S_k1 = [(k1_var - k1) * 0.5 for k1_var in k1_values]
S_k2 = [(k2_var - k2) * (-2.0) for k2_var in k2_values]

ax3.plot(k1_values, S_k1, 'b-', label='k₁ sensitivity', linewidth=2)
ax3.plot(k2_values, S_k2, 'r-', label='k₂ sensitivity', linewidth=2)

ax3.set_xlabel('Parameter Value')
ax3.set_ylabel('ΔT Change (hours)')
ax3.set_title('Parameter Sensitivity Analysis (Classical)')
ax3.legend()
ax3.grid(alpha=0.3)

# --- Subplot 4: Residual Analysis (Classical Model Fit to Simulated Observations) ---
ax4 = plt.subplot(2, 3, 4)
np.random.seed(42)
t_obs = np.linspace(0.5, 6.0, 25)  # Observed time points
A0_ref = calculate_initial_concentration(70, 0.68, 350, 5)

# True model predictions (classical, mg/dL)
B_pred = np.array([
    classical_bac_model(ti, A0_ref, k1, k2)[1] * 1000
    for ti in t_obs
])

# Simulate “observed” data with heteroscedastic noise
noise_scale = 0.003 + 0.002 * (B_pred / B_pred.max())
B_obs = B_pred + np.random.normal(0, noise_scale)

# Residuals
residuals = B_obs - B_pred

ax4.scatter(t_obs, residuals, color='tab:green', s=50, alpha=0.7)
ax4.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Add trend line for residuals
coeffs = np.polyfit(t_obs, residuals, 1)
trend  = np.poly1d(coeffs)(t_obs)
ax4.plot(t_obs, trend, color='red', linewidth=1, alpha=0.5)

ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Residual (mg/dL)')
ax4.set_title('Residual Analysis (Classical Model)')
ax4.text(
    0.02, 0.95,
    f'RMSE = {np.sqrt(np.mean(residuals**2)):.4f} mg/dL',
    transform=ax4.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)
ax4.grid(alpha=0.3)

# --- Subplot 5: Effect of Fractional Order α (Simplified Comparison) ---
ax5 = plt.subplot(2, 3, 5)
alpha_vals = [0.6, 0.8, 1.0]
colors     = ['tab:blue', 'tab:green', 'tab:red']

for α_val, col in zip(alpha_vals, colors):
    B_cur = np.zeros_like(t_grid)
    for i, t in enumerate(t_grid):
        if abs(α_val - 1.0) < 1e-12:
            # Classical model when α = 1
            _, B_cl = classical_bac_model(t, A0_ref, k1, k2)
            B_cur[i] = B_cl * 1000  # g/dL → mg/dL
        else:
            # Simplified “fractional” approximation: scale k1 by α
            _, B_cl = classical_bac_model(t, A0_ref, k1 * α_val, k2)
            B_cur[i] = B_cl * 1000
    ax5.plot(
        t_grid, B_cur, color=col, linewidth=2,
        label=f'α = {α_val}' + (' (Classical)' if α_val == 1.0 else ' (Frac approx.)')
    )

ax5.axhline(y=BAC_high, color='red', linestyle='--', alpha=0.5)
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('BAC (mg/dL)')
ax5.set_title('Effect of Fractional Order α (Approx.)')
ax5.legend(fontsize='small')
ax5.grid(alpha=0.3)

# --- Subplot 6: Peak BAC by Gender & Weight (Classical Model) ---
ax6 = plt.subplot(2, 3, 6)
weights_list = [50, 60, 70, 80, 90]
male_peaks   = []
female_peaks = []

for w in weights_list:
    # Male (r = 0.68)
    A0_m  = calculate_initial_concentration(w, 0.68, 350, 5)
    B_m   = np.array([classical_bac_model(ti, A0_m, k1, k2)[1] * 1000 for ti in t_grid])
    male_peaks.append(B_m.max())

    # Female (r = 0.55)
    A0_f  = calculate_initial_concentration(w, 0.55, 350, 5)
    B_f   = np.array([classical_bac_model(ti, A0_f, k1, k2)[1] * 1000 for ti in t_grid])
    female_peaks.append(B_f.max())

x = np.arange(len(weights_list))
width = 0.35

ax6.bar(x - width/2, male_peaks, width, label='Male',   color='tab:blue', alpha=0.7)
ax6.bar(x + width/2, female_peaks, width, label='Female', color='tab:red',   alpha=0.7)
ax6.axhline(y=BAC_high, color='red', linestyle='--', alpha=0.7)

ax6.set_xlabel('Body Weight (kg)')
ax6.set_ylabel('Peak BAC (mg/dL)')
ax6.set_title('Peak BAC by Gender & Weight (Classical)')
ax6.set_xticks(x)
ax6.set_xticklabels(weights_list)
ax6.legend(fontsize='small')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Example Summary Statistics
# ──────────────────────────────────────────────────────────────────────────────

print("=== Blood Alcohol Concentration Model Results ===")
print("Model parameters:")
print(f"  Absorption rate k₁        = {k1}")
print(f"  Elimination rate k₂       = {k2}")
print(f"  Fractional orders (α, β)  = ({alpha}, {beta})\n")

# Example: 70 kg male drinking 350 mL beer (5 % ABV)
A0_ex  = calculate_initial_concentration(70, 0.68, 350, 5)
print("Example: 70 kg male drinking 350 mL beer (5 % ABV)")
print(f"  Initial concentration A₀ = {A0_ex:.4f} g/L")

# Compute classical B(t) for summary
B_ex = np.array([classical_bac_model(ti, A0_ex, k1, k2)[1] * 1000 for ti in t_grid])
peak_idx = B_ex.argmax()
peak_time = t_grid[peak_idx]
peak_bac = B_ex[peak_idx]

print(f"  Peak BAC           = {peak_bac:.2f} mg/dL at t = {peak_time:.2f} h")

t_i_ex = find_threshold_time(B_ex, t_grid, BAC_high)
t_f_ex = None
idx_above_low_ex = np.where(B_ex >= BAC_low)[0]
if idx_above_low_ex.size > 0:
    t_f_ex = t_grid[idx_above_low_ex[-1]]

if t_i_ex is not None:
    print(f"  Time to 0.08 mg/dL  = {t_i_ex:.2f} h")
if t_f_ex is not None:
    ΔT_ex = t_f_ex - t_i_ex if t_i_ex is not None else (t_f_ex - 0.0)
    print(f"  Time to 0.01 mg/dL  = {t_f_ex:.2f} h")
    print(f"  Tolerance ΔT        = {ΔT_ex:.2f} h")
else:
    print("  BAC never exceeded 0.01 mg/dL (no tolerance interval)")

