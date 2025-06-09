import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Simplified test of the corrected fractional BAC model
def ml1_simple(z, alpha, terms=30):
    """Simplified Mittag-Leffler function for testing"""
    if abs(z) > 10:
        return 0 if z < 0 else np.inf
    
    result = 0
    for n in range(terms):
        try:
            term = (z**n) / gamma(alpha*n + 1)
            if abs(term) < 1e-12:
                break
            result += term
        except:
            break
    return result

def test_fractional_bac(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """Test version of fractional BAC model"""
    if t == 0:
        return A0, 0
    
    # Stomach concentration decreases
    A_t = A0 * ml1_simple(-k1 * (t**alpha), alpha)
    
    # Blood concentration with proper behavior
    if t > 0:
        # Simple corrected formulation
        term1 = ml1_simple(-k1 * (t**alpha), alpha)
        term2 = ml1_simple(-k2 * (t**beta), beta)
        
        if k2 > k1:
            B_t = (k1 * A0 / (k2 - k1)) * (term1 - term2)
        else:
            B_t = k1 * A0 * term1 * term2
        
        B_t = max(0, B_t * 0.1)  # Convert to mg/100mL
    else:
        B_t = 0
    
    return max(0, A_t), B_t

# Test the model behavior
print("Testing Corrected Fractional BAC Model")
print("="*50)

A0 = 3.45  # Initial concentration for 70kg male drinking beer
k1, k2 = 1.0, 0.12
alpha, beta = 0.8, 0.9

time_points = [0, 0.5, 1, 2, 3, 4, 6, 8, 10]
print("Time (hrs) | A(t) Stomach | B(t) Blood")
print("-" * 40)

for t in time_points:
    A_t, B_t = test_fractional_bac(t, A0, k1, k2, alpha, beta)
    print(f"{t:8.1f}   | {A_t:10.4f}   | {B_t:8.4f}")

print("\nKey Observations:")
print("1. A(t) should decrease monotonically from A0")
print("2. B(t) should increase to a peak then decrease")
print("3. Both should approach 0 as t → ∞")
print("4. Model shows memory effects with fractional orders < 1")