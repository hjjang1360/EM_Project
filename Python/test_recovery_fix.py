#!/usr/bin/env python3
"""
Test the fixed recovery time logic
수정된 회복 시간 로직 테스트
"""

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

# Configure Korean fonts for matplotlib
import matplotlib.font_manager as fm

try:
    font_list = [font.name for font in fm.fontManager.ttflist]
    korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Dotum"]
    available_font = None

    for font in korean_fonts:
        if font in font_list:
            available_font = font
            break

    if available_font:
        plt.rcParams["font.family"] = available_font
        print(f"✅ 한글 폰트 설정: {available_font}")
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        print("⚠️ 한글 폰트를 찾을 수 없음. 기본 폰트 사용.")

except:
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("⚠️ 폰트 설정 실패. 기본 폰트 사용.")

plt.rcParams["axes.unicode_minus"] = False


class TestBACCalculator:
    def __init__(self):
        self.k1, self.k2 = 0.8, 1.0
        self.alpha, self.beta = 0.8, 0.9

    def ml1_stable(self, z, alpha, max_terms=100, tol=1e-15):
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

    def fractional_bac_model_corrected(self, t, A0, k1, k2, alpha=0.8, beta=0.9):
        """Theoretically correct fractional BAC model"""
        if t == 0:
            return A0, 0.0

        if t < 0:
            return A0, 0.0

        # Stomach concentration A(t)
        A_t = A0 * self.ml1_stable(-k1 * (t**alpha), alpha)

        # Blood concentration B(t)
        if t > 0:
            if abs(k2 - k1) < 1e-10:
                B_t = A0 * k1 * (t**alpha) * self.ml1_stable(-k1 * (t**alpha), alpha)
            else:
                term1 = self.ml1_stable(-k1 * (t**alpha), alpha)
                term2 = self.ml1_stable(-k2 * (t**beta), beta)
                B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

            B_t = max(0.0, min(B_t, A0))
        else:
            B_t = 0.0

        return max(0.0, A_t), B_t

    def calculate_initial_concentration(self, weight, tbw_ratio, volume, abv):
        """Calculate initial alcohol concentration A0 in stomach"""
        rho_ethanol = 0.789  # g/mL
        alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
        tbw_volume = tbw_ratio * weight  # liters
        A0 = alcohol_mass / tbw_volume  # g/L
        return A0

    def find_recovery_times_old(self, t_array, bac_array):
        """OLD (PROBLEMATIC) recovery time calculation"""
        bac_mg = bac_array * 100  # Convert to mg/100mL

        # Legal driving limit (50 mg/100mL in Korea)
        legal_idx = np.where(bac_mg <= 50)[0]
        legal_time = t_array[legal_idx[0]] if len(legal_idx) > 0 else None

        # Safe driving limit (30 mg/100mL)
        safe_idx = np.where(bac_mg <= 30)[0]
        safe_time = t_array[safe_idx[0]] if len(safe_idx) > 0 else None

        # Complete recovery (10 mg/100mL)
        recovery_idx = np.where(bac_mg <= 10)[0]
        recovery_time = t_array[recovery_idx[0]] if len(recovery_idx) > 0 else None

        return legal_time, safe_time, recovery_time

    def find_recovery_times_new(self, t_array, bac_array):
        """NEW (FIXED) recovery time calculation"""
        bac_mg = bac_array * 100  # Convert to mg/100mL

        # Find peak BAC first to avoid catching initial zero values
        peak_idx = np.argmax(bac_mg)
        peak_time = t_array[peak_idx]
        peak_bac = bac_mg[peak_idx]

        # If peak BAC is too low, no meaningful recovery time calculation
        if peak_bac < 10:
            return None, None, None

        # Only look for recovery times after the peak
        post_peak_mask = t_array > peak_time

        if not np.any(post_peak_mask):
            return None, None, None

        post_peak_times = t_array[post_peak_mask]
        post_peak_bac = bac_mg[post_peak_mask]

        # Legal driving limit (50 mg/100mL in Korea)
        legal_idx = np.where(post_peak_bac <= 50)[0]
        legal_time = post_peak_times[legal_idx[0]] if len(legal_idx) > 0 else None

        # Safe driving limit (30 mg/100mL)
        safe_idx = np.where(post_peak_bac <= 30)[0]
        safe_time = post_peak_times[safe_idx[0]] if len(safe_idx) > 0 else None

        # Complete recovery (10 mg/100mL)
        recovery_idx = np.where(post_peak_bac <= 10)[0]
        recovery_time = (
            post_peak_times[recovery_idx[0]] if len(recovery_idx) > 0 else None
        )

        return legal_time, safe_time, recovery_time


def test_recovery_logic():
    """Test both old and new recovery time logic"""
    print("🧪 회복 시간 로직 테스트")
    print("=" * 50)

    calculator = TestBACCalculator()

    # Test scenario: 남성, 25세, 70kg, 소주 360mL
    weight = 70
    age = 25
    tbw_ratio = 0.68
    volume = 360
    abv = 17

    print(f"테스트 시나리오:")
    print(f"- 남성, {age}세, {weight}kg")
    print(f"- 소주 {volume}mL ({abv}%)")

    # Calculate initial concentration
    A0 = calculator.calculate_initial_concentration(weight, tbw_ratio, volume, abv)
    print(f"- 초기 농도 A0: {A0:.3f} g/L")

    # Time array
    t_hours = np.linspace(0, 24, 1000)
    bac_values = []

    for t in t_hours:
        _, B_t = calculator.fractional_bac_model_corrected(
            t, A0, calculator.k1, calculator.k2, calculator.alpha, calculator.beta
        )
        bac_values.append(B_t * 100)  # Convert to mg/100mL

    bac_array = np.array(bac_values) / 100  # Back to g/L for function

    # Test old logic
    print(f"\n📊 기존 로직 (문제있음):")
    old_legal, old_safe, old_recovery = calculator.find_recovery_times_old(
        t_hours, bac_array
    )
    print(
        f"- 운전 가능 (50mg/100mL): {old_legal:.1f}시간"
        if old_legal
        else "- 운전 가능: 24시간 내 불가능"
    )
    print(
        f"- 안전 운전 (30mg/100mL): {old_safe:.1f}시간"
        if old_safe
        else "- 안전 운전: 24시간 내 불가능"
    )
    print(
        f"- 완전 회복 (10mg/100mL): {old_recovery:.1f}시간"
        if old_recovery
        else "- 완전 회복: 24시간 내 불가능"
    )

    # Test new logic
    print(f"\n📊 개선된 로직 (수정됨):")
    new_legal, new_safe, new_recovery = calculator.find_recovery_times_new(
        t_hours, bac_array
    )
    print(
        f"- 운전 가능 (50mg/100mL): {new_legal:.1f}시간"
        if new_legal
        else "- 운전 가능: 24시간 내 불가능"
    )
    print(
        f"- 안전 운전 (30mg/100mL): {new_safe:.1f}시간"
        if new_safe
        else "- 안전 운전: 24시간 내 불가능"
    )
    print(
        f"- 완전 회복 (10mg/100mL): {new_recovery:.1f}시간"
        if new_recovery
        else "- 완전 회복: 24시간 내 불가능"
    )

    # Find peak info
    bac_mg = bac_array * 100
    peak_idx = np.argmax(bac_mg)
    peak_time = t_hours[peak_idx]
    peak_bac = bac_mg[peak_idx]

    print(f"\n📈 BAC 피크 정보:")
    print(f"- 최고 BAC: {peak_bac:.1f} mg/100mL")
    print(f"- 피크 시간: {peak_time:.1f}시간")

    # Create comparison plot
    plt.figure(figsize=(15, 10))

    # Plot 1: BAC curve with old logic
    plt.subplot(2, 1, 1)
    plt.plot(t_hours, bac_mg, "b-", linewidth=3, label="혈중알코올농도", alpha=0.8)

    # Add threshold lines
    plt.axhline(
        y=50, color="orange", linestyle="--", alpha=0.7, label="운전가능 (50mg/100mL)"
    )
    plt.axhline(
        y=30, color="green", linestyle="--", alpha=0.7, label="안전운전 (30mg/100mL)"
    )
    plt.axhline(
        y=10, color="blue", linestyle="--", alpha=0.7, label="완전회복 (10mg/100mL)"
    )

    # Mark old recovery times
    if old_legal:
        plt.axvline(
            x=old_legal,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"기존 운전가능: {old_legal:.1f}h",
        )
    if old_safe:
        plt.axvline(
            x=old_safe,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"기존 안전운전: {old_safe:.1f}h",
        )
    if old_recovery:
        plt.axvline(
            x=old_recovery,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"기존 완전회복: {old_recovery:.1f}h",
        )

    plt.axvline(
        x=peak_time,
        color="red",
        linestyle="-",
        linewidth=2,
        alpha=0.8,
        label=f"피크: {peak_time:.1f}h",
    )

    plt.xlabel("시간 (hours)")
    plt.ylabel("BAC (mg/100mL)")
    plt.title("기존 로직 - 문제: 초기 시점(t=0)을 잘못 인식")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)
    plt.ylim(0, max(100, np.max(bac_mg) * 1.1))

    # Plot 2: BAC curve with new logic
    plt.subplot(2, 1, 2)
    plt.plot(t_hours, bac_mg, "b-", linewidth=3, label="혈중알코올농도", alpha=0.8)

    # Add threshold lines
    plt.axhline(
        y=50, color="orange", linestyle="--", alpha=0.7, label="운전가능 (50mg/100mL)"
    )
    plt.axhline(
        y=30, color="green", linestyle="--", alpha=0.7, label="안전운전 (30mg/100mL)"
    )
    plt.axhline(
        y=10, color="blue", linestyle="--", alpha=0.7, label="완전회복 (10mg/100mL)"
    )

    # Mark new recovery times
    if new_legal:
        plt.axvline(
            x=new_legal,
            color="orange",
            linestyle=":",
            linewidth=3,
            label=f"개선된 운전가능: {new_legal:.1f}h",
        )
    if new_safe:
        plt.axvline(
            x=new_safe,
            color="green",
            linestyle=":",
            linewidth=3,
            label=f"개선된 안전운전: {new_safe:.1f}h",
        )
    if new_recovery:
        plt.axvline(
            x=new_recovery,
            color="blue",
            linestyle=":",
            linewidth=3,
            label=f"개선된 완전회복: {new_recovery:.1f}h",
        )

    plt.axvline(
        x=peak_time,
        color="red",
        linestyle="-",
        linewidth=2,
        alpha=0.8,
        label=f"피크: {peak_time:.1f}h",
    )

    # Mark peak area
    plt.axvspan(0, peak_time, alpha=0.1, color="red", label="피크 이전 (무시)")
    plt.axvspan(peak_time, 24, alpha=0.1, color="green", label="피크 이후 (고려)")

    plt.xlabel("시간 (hours)")
    plt.ylabel("BAC (mg/100mL)")
    plt.title("개선된 로직 - 피크 이후 시점만 고려하여 정확한 예측")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)
    plt.ylim(0, max(100, np.max(bac_mg) * 1.1))

    plt.tight_layout()
    plt.savefig("recovery_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n✅ 테스트 완료!")
    print(f"📊 그래프가 'recovery_time_comparison.png'로 저장되었습니다.")
    print(f"\n🔧 주요 개선 사항:")
    print(f"1. 피크 시간 이후에만 회복 시간을 계산")
    print(f"2. 초기 시점(t=0)의 BAC=0을 잘못 인식하는 문제 해결")
    print(f"3. 한글 폰트 지원으로 그래프 가독성 향상")


if __name__ == "__main__":
    test_recovery_logic()
