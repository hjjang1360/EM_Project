"""
Analysis of Fractional BAC Model Issues
Based on the PDF and LaTeX documents provided
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

print("=== FRACTIONAL BAC MODEL ISSUE ANALYSIS ===")
print()

# Issue 1: Mathematical Formulation Inconsistency
print("1. MATHEMATICAL FORMULATION ISSUES:")
print("   - Multiple different formulas used across versions")
print("   - Theoretical foundation from PDF:")
print("     A(t) = A₀ E_α(-k₁ t^α)")
print("     B(t) = (A₀ k₁)/(k₂ - k₁) [E_α(-k₁ t^α) - E_β(-k₂ t^β)]")
print()
print("   - But implementations use various formulas:")
print("     • all_v7/v8: B(t) = k₁ A₀ t^(β-1) E_{α,β}(-k₁t^α, -k₂t^β)")
print("     • all_v9: Multiple different attempts")
print()

# Issue 2: Numerical Implementation Problems
print("2. NUMERICAL IMPLEMENTATION PROBLEMS:")


def problematic_mittag_leffler(z, alpha, n_terms=50):
    """Demonstrates the numerical issues in current implementation"""
    result = 0
    for n in range(n_terms):
        try:
            term = (z**n) / gamma(alpha * n + 1)
            result += term
        except (OverflowError, RuntimeError):
            return float("inf")  # This causes issues
    return result


# Test with problematic values
test_values = [-10, -5, -1, 0, 1, 5, 10]
print("   Current Mittag-Leffler function issues:")
for z in test_values:
    try:
        result = problematic_mittag_leffler(z, 0.8)
        print(f"     E_0.8({z:3.0f}) = {result:10.3f}")
    except:
        print(f"     E_0.8({z:3.0f}) = ERROR")
print()

# Issue 3: Parameter Inconsistencies
print("3. PARAMETER INCONSISTENCIES ACROSS FILES:")
parameters = [
    ("all_v7.py", "k1=1.0, k2=0.12"),
    ("all_v9.py", "k1=1.0, k2=1.2"),
    ("plot_fixed_v2_copy.py", "k1=0.8, k2=1.0"),
    ("LaTeX docs", "k1=0.8, k2=1.0 (recommended)"),
]

for file, params in parameters:
    print(f"   {file:20s}: {params}")
print()

# Issue 4: Unit Conversion Problems
print("4. UNIT CONVERSION INCONSISTENCIES:")
print("   - Some functions multiply by 0.1 (g/L → g/100mL)")
print("   - Others multiply by 100 (g/L → mg/100mL)")
print("   - Inconsistent application leads to wrong scales")
print()

# Issue 5: Physical Unrealistic Behavior
print("5. PHYSICAL REALISM ISSUES:")
print("   - BAC doesn't always decrease monotonically after peak")
print("   - Some implementations give negative BAC values")
print("   - Peak times and values often unrealistic")
print("   - Recovery times don't match physiological expectations")
print()

# Issue 6: Fallback Mechanisms
print("6. PROBLEMATIC FALLBACK MECHANISMS:")
print("   Examples from all_v9.py:")
print("   • if B_t <= 0 or not np.isfinite(B_t):")
print("     B_t = A0 * 0.05 * np.exp(-k2 * t) / (1 + t)")
print("   This arbitrary fallback undermines the fractional model's purpose")
print()

print("=== RECOMMENDED FIXES ===")
print()
print("1. Use theoretically correct formula from comprehensive LaTeX docs")
print("2. Implement numerically stable Mittag-Leffler functions")
print("3. Standardize parameters: k1=0.8, k2=1.0, α=0.8, β=0.9")
print("4. Consistent unit handling: work in g/L, convert to mg/100mL for display")
print("5. Add physical constraints: 0 ≤ B(t) ≤ A₀, monotonic decrease after peak")
print("6. Remove arbitrary fallback mechanisms")
print()

print("=== SPECIFIC PROBLEMS IN CURRENT FILES ===")
print()

# Analysis of specific implementations
implementations = {
    "all_v2.py to all_v6.py": [
        "Uses oversimplified B(t) = k₁ A₀ t^(β-1) E^(2)_{α,β}(-k₁t^α, -k₂t^β)",
        "This formula doesn't match theoretical foundation",
        "Doesn't ensure proper elimination kinetics",
    ],
    "all_v7.py, all_v8.py": [
        "Same mathematical error as above",
        "Better numerical implementation but wrong formula",
    ],
    "all_v9.py": [
        "Multiple contradictory implementations in same file",
        "Numerous fallback mechanisms that break fractional behavior",
        "Inconsistent parameter choices",
        "Complex peak-finding logic that's often wrong",
    ],
}

for impl, issues in implementations.items():
    print(f"{impl}:")
    for issue in issues:
        print(f"   • {issue}")
    print()

print("=== CONCLUSION ===")
print("The main issue is using incorrect mathematical formulations")
print("that don't match the theoretical foundation described in the PDF.")
print("The correct approach should follow the Caputo derivative solutions:")
print()
print("A(t) = A₀ E_α(-k₁ t^α)")
print("B(t) = (A₀ k₁)/(k₂ - k₁) [E_α(-k₁ t^α) - E_β(-k₂ t^β)]")
print()
print("With numerically stable implementations and consistent parameters.")
