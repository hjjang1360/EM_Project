import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Mittag-Leffler functions
def mittag_leffler(z, alpha, n_terms=50):
    result = 0.0
    for n in range(n_terms):
        term = z**n / gamma(alpha*n + 1)
        result += term
        if abs(term) < 1e-12:
            break
    return result

def two_param_mittag_leffler(x, y, alpha, beta, n_terms=20):
    result = 0.0
    for m in range(n_terms):
        for n in range(n_terms):
            result += (x**m * y**n) / gamma(alpha*m + beta*n + 1)
    return result

# Model functions
def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv/100) * rho_ethanol
    tbw_volume = tbw_ratio * weight
    return alcohol_mass / tbw_volume

def classical_bac_model(t, A0, k1, k2):
    if t == 0:
        return 0.0
    if k1 != k2:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1*t) - np.exp(-k2*t)) / 10
    else:
        B_t = k1 * A0 * t * np.exp(-k1*t) / 10
    return B_t

def fractional_bac_model(t, A0, k1, k2, alpha=0.8, beta=0.9):
    if t == 0:
        return 0.0
    B_t = k1 * A0 * t**(beta-1) * two_param_mittag_leffler(-k1*t**alpha, -k2*t**beta, alpha, beta)
    return B_t / 10

# Parameters
scenarios = [
    ("Low ABV Male", 65, 0.68, 350, 5),
    ("Low ABV Female", 65, 0.55, 350, 5),
    ("High ABV Male", 65, 0.68, 50, 40),
    ("High ABV Female", 65, 0.55, 50, 40),
]
k1, k2 = 1.2, 0.15
alpha, beta = 0.8, 0.9
t = np.linspace(0, 8, 200)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Classic BAC vs Time
ax = axes[0, 0]
for label, w, r, vol, abv in scenarios:
    A0 = calculate_initial_concentration(w, r, vol, abv)
    B_vals = [classical_bac_model(time, A0, k1, k2) for time in t]
    ax.plot(t, B_vals, label=label)
ax.set_xlabel("Time (h)")
ax.set_ylabel("BAC (%)")
ax.set_title("Classic BAC decay vs Time")
ax.legend()
ax.grid(True)

# 2. Classic Recovery Time vs TBW
ax = axes[0, 1]
tbw_vals = [r*w for _, w, r, _, _ in scenarios]
recovery_times = []
for _, w, r, vol, abv in scenarios:
    A0 = calculate_initial_concentration(w, r, vol, abv)
    B_vals = np.array([classical_bac_model(time, A0, k1, k2) for time in t])
    peak_idx = np.argmax(B_vals)
    rec_idx = np.where((t > t[peak_idx]) & (B_vals <= 0.01))[0]
    rec_time = t[rec_idx[0]] if len(rec_idx) > 0 else np.nan
    recovery_times.append(rec_time)
ax.plot(tbw_vals, recovery_times, marker="o")
ax.set_xlabel("Total Body Water (L)")
ax.set_ylabel("Time to BAC<0.01% (h)")
ax.set_title("Classic Recovery Time vs TBW")
ax.grid(True)

# 3. Classic Recovery Time vs Weight
ax = axes[0, 2]
weights = np.arange(60, 105, 5)
rec_times_w = []
for w in weights:
    A0 = calculate_initial_concentration(w, 0.68, 350, 5)
    B_vals = np.array([classical_bac_model(time, A0, k1, k2) for time in t])
    peak_idx = np.argmax(B_vals)
    rec_idx = np.where((t > t[peak_idx]) & (B_vals <= 0.01))[0]
    rec_time = t[rec_idx[0]] if len(rec_idx) > 0 else np.nan
    rec_times_w.append(rec_time)
ax.plot(weights, rec_times_w, marker="o")
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Time to BAC<0.01% (h)")
ax.set_title("Classic Recovery Time vs Weight")
ax.grid(True)

# 4. Fractional BAC vs Time
ax = axes[1, 0]
for label, w, r, vol, abv in scenarios:
    A0 = calculate_initial_concentration(w, r, vol, abv)
    B_vals = [fractional_bac_model(time, A0, k1, k2, alpha, beta) for time in t]
    ax.plot(t, B_vals, label=label)
ax.set_xlabel("Time (h)")
ax.set_ylabel("BAC (%)")
ax.set_title("Fractional BAC decay vs Time")
ax.legend()
ax.grid(True)

# 5. Fractional Recovery Time vs TBW
ax = axes[1, 1]
recovery_times_frac = []
for _, w, r, vol, abv in scenarios:
    A0 = calculate_initial_concentration(w, r, vol, abv)
    B_vals = np.array([fractional_bac_model(time, A0, k1, k2, alpha, beta) for time in t])
    peak_idx = np.argmax(B_vals)
    rec_idx = np.where((t > t[peak_idx]) & (B_vals <= 0.01))[0]
    rec_time = t[rec_idx[0]] if len(rec_idx) > 0 else np.nan
    recovery_times_frac.append(rec_time)
ax.plot(tbw_vals, recovery_times_frac, marker="o")
ax.set_xlabel("Total Body Water (L)")
ax.set_ylabel("Time to BAC<0.01% (h)")
ax.set_title("Fractional Recovery Time vs TBW")
ax.grid(True)

# 6. Fractional Recovery Time vs Weight
ax = axes[1, 2]
rec_times_w_frac = []
for w in weights:
    A0 = calculate_initial_concentration(w, 0.68, 350, 5)
    B_vals = np.array([fractional_bac_model(time, A0, k1, k2, alpha, beta) for time in t])
    peak_idx = np.argmax(B_vals)
    rec_idx = np.where((t > t[peak_idx]) & (B_vals <= 0.01))[0]
    rec_time = t[rec_idx[0]] if len(rec_idx) > 0 else np.nan
    rec_times_w_frac.append(rec_time)
ax.plot(weights, rec_times_w_frac, marker="o")
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Time to BAC<0.01% (h)")
ax.set_title("Fractional Recovery Time vs Weight")
ax.grid(True)

plt.tight_layout()
plt.show()
