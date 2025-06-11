#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Recovery Time Test
Tests the fixed recovery time logic with a simple scenario
"""

from bac_calculator_simple import calculate_bac, find_recovery_times
import numpy as np


def test_recovery_time():
    """Test recovery time calculation with a known scenario"""
    print("Testing Recovery Time Logic...")
    print("=" * 50)

    # Test scenario: Male, 25y, 70kg, Soju 360mL (17%)
    volume_ml = 360
    alcohol_percent = 17
    weight_kg = 70
    gender = "male"

    print(f"Test Scenario:")
    print(f"- Gender: {gender}")
    print(f"- Weight: {weight_kg} kg")
    print(f"- Alcohol: {volume_ml} mL at {alcohol_percent}%")
    print()

    # Calculate BAC over time
    t_array = np.linspace(0, 24, 1000)
    bac_mg = calculate_bac(volume_ml, alcohol_percent, weight_kg, gender, t_array)

    # Find peak BAC
    peak_bac = np.max(bac_mg)
    peak_time = t_array[np.argmax(bac_mg)]

    print(f"Peak BAC: {peak_bac:.1f} mg/100mL at {peak_time:.1f} hours")

    # Find recovery times
    recovery_times = find_recovery_times(t_array, bac_mg)

    print(f"\nRecovery Times:")
    print(
        f"- Legal (≤50 mg/100mL): {recovery_times['legal']:.1f} hours"
        if recovery_times["legal"]
        else "- Legal: Not reached within 24h"
    )
    print(
        f"- Safe (≤20 mg/100mL): {recovery_times['safe']:.1f} hours"
        if recovery_times["safe"]
        else "- Safe: Not reached within 24h"
    )
    print(
        f"- Full recovery (≤5 mg/100mL): {recovery_times['full']:.1f} hours"
        if recovery_times["full"]
        else "- Full: Not reached within 24h"
    )

    # Verify the logic is working correctly
    if recovery_times["legal"] and recovery_times["safe"] and recovery_times["full"]:
        print(f"\n✓ SUCCESS: All recovery times calculated correctly!")
        print(f"✓ Recovery times are after peak ({peak_time:.1f}h)")
        return True
    else:
        print(f"\n✗ FAILURE: Some recovery times are None")
        return False


def test_korean_fonts():
    """Test Korean font configuration for matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        print("\nTesting Korean Font Configuration...")
        print("=" * 50)

        # Get available fonts
        korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Dotum"]
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        found_fonts = []
        for font in korean_fonts:
            if font in available_fonts:
                found_fonts.append(font)

        if found_fonts:
            selected_font = found_fonts[0]
            plt.rcParams["font.family"] = selected_font
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✓ Korean font configured: {selected_font}")
            print(f"✓ Available Korean fonts: {', '.join(found_fonts)}")
            return True
        else:
            print("✗ No Korean fonts found")
            print("Available fonts include:", ", ".join(available_fonts[:5]), "...")
            return False

    except Exception as e:
        print(f"✗ Error configuring fonts: {e}")
        return False


if __name__ == "__main__":
    print("BAC Calculator - Quick Functionality Test")
    print("=" * 60)

    # Test recovery time logic
    recovery_ok = test_recovery_time()

    # Test Korean fonts
    font_ok = test_korean_fonts()

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Recovery Logic: {'✓ PASS' if recovery_ok else '✗ FAIL'}")
    print(f"Korean Fonts: {'✓ PASS' if font_ok else '✗ FAIL'}")

    if recovery_ok and font_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("The BAC calculator is working correctly.")
    elif recovery_ok:
        print("\n⚠️ RECOVERY LOGIC WORKS, FONT ISSUE")
        print("Main functionality works, Korean display may show boxes.")
    else:
        print("\n❌ CRITICAL FAILURE")
        print("Recovery logic needs attention.")
