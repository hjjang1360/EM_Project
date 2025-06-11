#!/usr/bin/env python3
"""
Quick validation script for BAC calculator functionality
"""

import numpy as np
from scipy.special import gamma


def ml1_stable(z, alpha, max_terms=100, tol=1e-15):
    """Numerically stable Mittag-Leffler function E_α(z)"""
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
    """Test fractional BAC model"""
    if t == 0:
        return A0, 0.0

    if t < 0:
        return A0, 0.0

    # Stomach concentration A(t)
    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    # Blood concentration B(t)
    if t > 0:
        if abs(k2 - k1) < 1e-10:
            B_t = A0 * k1 * (t**alpha) * ml1_stable(-k1 * (t**alpha), alpha)
        else:
            term1 = ml1_stable(-k1 * (t**alpha), alpha)
            term2 = ml1_stable(-k2 * (t**beta), beta)
            B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

        B_t = max(0.0, min(B_t, A0))
    else:
        B_t = 0.0

    return max(0.0, A_t), B_t


def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration A0 in stomach"""
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
    tbw_volume = tbw_ratio * weight  # liters
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0


def main():
    print("🧪 BAC Calculator Quick Test")
    print("=" * 40)

    # Test parameters (typical Korean male, soju consumption)
    weight = 70  # kg
    age = 25
    tbw_ratio = 0.68  # Male
    volume = 360  # mL (1 bottle of soju)
    abv = 17  # %

    print(f"Test scenario:")
    print(f"- 남성, {age}세, {weight}kg")
    print(f"- 소주 {volume}mL ({abv}%)")
    print(f"- 체수분 비율: {tbw_ratio:.2f}")

    # Calculate initial concentration
    A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)
    alcohol_mass = volume * abv / 100 * 0.789

    print(f"\n📊 계산 결과:")
    print(f"- 순수 알코올: {alcohol_mass:.1f}g")
    print(f"- 초기 농도 A0: {A0:.3f} g/L")

    # Test at different time points
    test_times = [0, 0.5, 1, 2, 4, 8, 12]

    print(f"\n⏰ 시간별 BAC 예측:")
    print("시간(h) | BAC(mg/100mL) | 상태")
    print("-" * 35)

    for t in test_times:
        A_t, B_t = fractional_bac_model(t, A0)
        bac_mg = B_t * 100  # Convert to mg/100mL

        # Determine status
        if bac_mg >= 80:
            status = "음주운전 단속"
        elif bac_mg >= 50:
            status = "면허정지"
        elif bac_mg >= 30:
            status = "운전 위험"
        elif bac_mg >= 10:
            status = "운전 가능"
        else:
            status = "완전 회복"

        print(f"{t:6.1f}  |  {bac_mg:8.1f}     | {status}")

    # Find recovery times
    t_array = np.linspace(0, 24, 1000)
    bac_values = []

    for t in t_array:
        _, B_t = fractional_bac_model(t, A0)
        bac_values.append(B_t * 100)  # mg/100mL

    bac_array = np.array(bac_values)

    # Find when BAC drops below thresholds
    legal_idx = np.where(bac_array <= 50)[0]
    safe_idx = np.where(bac_array <= 30)[0]
    recovery_idx = np.where(bac_array <= 10)[0]

    legal_time = t_array[legal_idx[0]] if len(legal_idx) > 0 else None
    safe_time = t_array[safe_idx[0]] if len(safe_idx) > 0 else None
    recovery_time = t_array[recovery_idx[0]] if len(recovery_idx) > 0 else None

    peak_bac = np.max(bac_array)
    peak_time = t_array[np.argmax(bac_array)]

    print(f"\n📈 요약:")
    print(f"- 최고 BAC: {peak_bac:.1f} mg/100mL")
    print(f"- 최고점 도달: {peak_time:.1f}시간 후")

    print(f"\n⏰ 회복 시간:")
    if legal_time:
        print(f"🚗 운전 가능 (50mg/100mL): {legal_time:.1f}시간 후")
    else:
        print("🚗 운전 가능: 24시간 내 불가능")

    if safe_time:
        print(f"✅ 안전 운전 (30mg/100mL): {safe_time:.1f}시간 후")
    else:
        print("✅ 안전 운전: 24시간 내 불가능")

    if recovery_time:
        print(f"🎉 완전 회복 (10mg/100mL): {recovery_time:.1f}시간 후")
    else:
        print("🎉 완전 회복: 24시간 내 불가능")

    print(f"\n⚠️ 주의: 이는 이론적 계산이며, 실제로는 개인차가 있을 수 있습니다.")
    print("실제 음주운전은 절대 금지이며, 음주 후에는 대중교통을 이용하세요!")

    print(f"\n✅ 계산기 테스트 완료!")


if __name__ == "__main__":
    main()
