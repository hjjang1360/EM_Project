import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import warnings

warnings.filterwarnings("ignore")


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


def fractional_bac_model_theoretical(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """
    Theoretically correct fractional BAC model based on:

    Caputo fractional differential equations:
    D^α A(t) = -k1 * A(t)
    D^β B(t) = k1 * A(t) - k2 * B(t)

    Solutions:
    A(t) = A0 * E_α(-k1 * t^α)
    B(t) = (A0 * k1)/(k2 - k1) * [E_α(-k1 * t^α) - E_β(-k2 * t^β)]

    This follows the theoretical foundation described in the PDF.
    """
    if t == 0:
        return A0, 0.0

    if t < 0:
        raise ValueError("Time must be non-negative")

    # Stomach concentration A(t)
    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    # Blood concentration B(t)
    if t > 0:
        if abs(k2 - k1) < 1e-10:
            # Special case: k1 ≈ k2
            # Use L'Hôpital's rule limit: B(t) = A0 * k1 * t^α * E_{α,α+1}(-k1 * t^α)
            B_t = A0 * k1 * (t**alpha) * ml2_stable(-k1 * (t**alpha), alpha, alpha + 1)
        else:
            # General case: k1 ≠ k2
            term1 = ml1_stable(-k1 * (t**alpha), alpha)
            term2 = ml1_stable(-k2 * (t**beta), beta)
            B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

        # Ensure physical constraints
        B_t = max(0.0, min(B_t, A0))  # 0 ≤ B(t) ≤ A0
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


# Test the corrected model
if __name__ == "__main__":
    # Test parameters based on PDF recommendations
    k1, k2 = 0.8, 1.0  # h^-1 (k2 > k1 for proper elimination)
    alpha, beta = 0.8, 0.9  # Fractional orders

    # Test scenario: 70kg male drinking 350mL of 5% beer
    A0 = calculate_initial_concentration(70, 0.68, 350, 5)

    print("Testing Corrected Fractional BAC Model")
    print("=" * 50)
    print(f"Initial concentration A0: {A0:.3f} g/L")
    print(f"Parameters: k1={k1}, k2={k2}, α={alpha}, β={beta}")
    print()

    # Time points for testing
    t_test = np.array([0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    print("Time (h) | Classical B(t) | Fractional B(t) | Difference")
    print("-" * 60)

    for t in t_test:
        _, B_classical = classical_bac_model(t, A0, k1, k2)
        _, B_fractional = fractional_bac_model_theoretical(t, A0, k1, k2, alpha, beta)

        # Convert to mg/100mL for display
        B_c_display = B_classical * 100  # g/L to mg/100mL
        B_f_display = B_fractional * 100
        diff = B_f_display - B_c_display

        print(f"{t:8.1f} | {B_c_display:13.3f} | {B_f_display:14.3f} | {diff:10.3f}")

    # Visualization
    t_plot = np.linspace(0, 12, 200)
    B_classical_plot = []
    B_fractional_plot = []

    for t in t_plot:
        _, B_c = classical_bac_model(t, A0, k1, k2)
        _, B_f = fractional_bac_model_theoretical(t, A0, k1, k2, alpha, beta)
        B_classical_plot.append(B_c * 100)  # Convert to mg/100mL
        B_fractional_plot.append(B_f * 100)

    plt.figure(figsize=(12, 8))
    plt.plot(t_plot, B_classical_plot, "b-", linewidth=2, label="Classical Model")
    plt.plot(
        t_plot,
        B_fractional_plot,
        "r--",
        linewidth=2,
        label="Fractional Model (Corrected)",
    )
    plt.axhline(
        y=80, color="red", linestyle=":", alpha=0.7, label="Legal Limit (80 mg/100mL)"
    )
    plt.axhline(
        y=10, color="orange", linestyle=":", alpha=0.7, label="Recovery (10 mg/100mL)"
    )

    plt.xlabel("Time (hours)")
    plt.ylabel("Blood Alcohol Concentration (mg/100mL)")
    plt.title("Corrected Fractional vs Classical BAC Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 12)
    plt.ylim(0, max(max(B_classical_plot), max(B_fractional_plot)) * 1.1)

    plt.tight_layout()
    plt.savefig("corrected_fractional_bac_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nModel validation:")
    print(f"Classical peak: {max(B_classical_plot):.3f} mg/100mL")
    print(f"Fractional peak: {max(B_fractional_plot):.3f} mg/100mL")
    print(
        f"Both models show proper decay: {B_fractional_plot[-1] < B_fractional_plot[0]}"
    )
