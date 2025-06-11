#!/usr/bin/env python3
"""
Demo test script to verify BAC calculator functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))


def test_simple_calculator():
    """Test the simple BAC calculator with predefined inputs"""
    try:
        from bac_calculator_simple import SimpleBACCalculator

        print("🧪 BAC Calculator Demo Test")
        print("=" * 40)

        # Create calculator instance
        calculator = SimpleBACCalculator()

        # Test with sample data
        print("Testing with sample data:")
        print("- 남성, 25세, 70kg")
        print("- 소주 360mL (17%)")
        print("- 분수계 모델")

        # Calculate TBW ratio for male, age 25
        tbw_ratio = 0.68 - (25 - 25) * 0.001  # = 0.68

        # Calculate initial concentration for soju
        volume = 360  # mL
        abv = 17  # %
        weight = 70  # kg
        A0 = calculator.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

        print(f"\n📊 계산 결과:")
        print(f"• 체수분 비율: {tbw_ratio:.2f}")
        print(f"• 초기 농도 (A0): {A0:.3f} g/L")
        print(f"• 순수 알코올: {volume * abv / 100 * 0.789:.1f}g")

        # Test fractional model at different time points
        import numpy as np

        test_times = [0, 0.5, 1, 2, 4, 8, 12]

        print(f"\n⏰ 시간별 BAC 예측 (분수계 모델):")
        for t in test_times:
            A_t, B_t = calculator.fractional_bac_model_corrected(
                t, A0, calculator.k1, calculator.k2, calculator.alpha, calculator.beta
            )
            bac_mg = B_t * 100  # Convert to mg/100mL
            print(f"• {t:2.1f}시간: {bac_mg:5.1f} mg/100mL")

        # Find recovery times
        t_hours = np.linspace(0, 24, 1000)
        bac_values = []

        for t in t_hours:
            _, B_t = calculator.fractional_bac_model_corrected(
                t, A0, calculator.k1, calculator.k2, calculator.alpha, calculator.beta
            )
            bac_values.append(B_t)

        bac_array = np.array(bac_values)
        legal_time, safe_time, recovery_time = calculator.find_recovery_times(
            t_hours, bac_array
        )

        # Display recovery predictions
        print(f"\n⏰ 회복 시간 예측:")
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

        print(f"\n✅ 테스트 완료! 계산기가 정상적으로 작동합니다.")
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gui_import():
    """Test if GUI calculator can be imported"""
    try:
        print("\n🖥️ GUI Calculator Import Test")
        print("=" * 40)

        import tkinter as tk
        from bac_calculator_gui import BACCalculatorApp

        print("✅ GUI 모듈 import 성공!")
        print("GUI 애플리케이션을 실행하려면 다음 명령을 사용하세요:")
        print("python bac_calculator_gui.py")

        return True

    except Exception as e:
        print(f"❌ GUI import 실패: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 BAC Calculator 종합 테스트")
    print("=" * 50)

    # Test simple calculator
    test1_result = test_simple_calculator()

    # Test GUI import
    test2_result = test_gui_import()

    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약:")
    print(f"• 간단한 계산기: {'✅ 성공' if test1_result else '❌ 실패'}")
    print(f"• GUI 모듈 import: {'✅ 성공' if test2_result else '❌ 실패'}")

    if test1_result and test2_result:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("\n사용 가능한 애플리케이션:")
        print("1. python bac_calculator_simple.py  # 명령줄 버전")
        print("2. python bac_calculator_gui.py     # GUI 버전")
        print("3. streamlit run bac_calculator_app.py  # 웹 버전")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")


if __name__ == "__main__":
    main()
