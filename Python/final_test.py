#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Testing Script for BAC Calculator Applications
Tests all fixed applications for syntax errors and basic functionality
"""

import sys
import os
import traceback


def test_import(module_name):
    """Test if a module can be imported without errors"""
    try:
        __import__(module_name.replace(".py", ""))
        return True, "SUCCESS"
    except Exception as e:
        return False, str(e)


def test_recovery_logic():
    """Test the fixed recovery time logic"""
    try:
        from bac_calculator_simple import calculate_bac, find_recovery_times
        import numpy as np

        # Test scenario: Male, 25y, 70kg, Soju 360mL (17%)
        volume_ml = 360
        alcohol_percent = 17
        weight_kg = 70
        gender = "male"

        # Calculate BAC
        t_array = np.linspace(0, 24, 1000)
        bac_mg = calculate_bac(volume_ml, alcohol_percent, weight_kg, gender, t_array)

        # Find recovery times
        recovery_times = find_recovery_times(t_array, bac_mg)

        # Check if recovery times are reasonable
        if (
            recovery_times["legal"]
            and recovery_times["safe"]
            and recovery_times["full"]
        ):
            return (
                True,
                f"Legal: {recovery_times['legal']:.1f}h, Safe: {recovery_times['safe']:.1f}h, Full: {recovery_times['full']:.1f}h",
            )
        else:
            return False, f"Some recovery times are None: {recovery_times}"

    except Exception as e:
        return False, str(e)


def test_korean_fonts():
    """Test Korean font configuration"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # Try to configure Korean fonts
        korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Dotum"]
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        korean_font_found = None
        for font in korean_fonts:
            if font in available_fonts:
                korean_font_found = font
                break

        if korean_font_found:
            plt.rcParams["font.family"] = korean_font_found
            plt.rcParams["axes.unicode_minus"] = False
            return True, f"Korean font configured: {korean_font_found}"
        else:
            return False, "No Korean fonts found"

    except Exception as e:
        return False, str(e)


def main():
    """Run all tests"""
    print("=" * 60)
    print("BAC Calculator - Final Testing Report")
    print("=" * 60)

    # List of files to test
    test_files = [
        "bac_calculator_simple.py",
        "bac_calculator_gui.py",
        "bac_calculator_enhanced.py",
        "bac_calculator_app.py",
        "bac_calculator_web.py",
        "bac_calculator_web_fixed.py",
    ]

    print("\n1. SYNTAX AND IMPORT TESTS")
    print("-" * 40)

    import_results = []
    for file in test_files:
        if os.path.exists(file):
            success, message = test_import(file)
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{file:<30} {status}")
            if not success:
                print(f"   Error: {message}")
            import_results.append((file, success))
        else:
            print(f"{file:<30} ✗ FILE NOT FOUND")
            import_results.append((file, False))

    print("\n2. RECOVERY LOGIC TEST")
    print("-" * 40)

    success, message = test_recovery_logic()
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"Recovery Time Logic           {status}")
    print(f"   Result: {message}")

    print("\n3. KOREAN FONT TEST")
    print("-" * 40)

    success, message = test_korean_fonts()
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"Korean Font Configuration     {status}")
    print(f"   Result: {message}")

    print("\n4. SUMMARY")
    print("-" * 40)

    passed_imports = sum(1 for _, success in import_results if success)
    total_imports = len(import_results)

    print(f"Import Tests: {passed_imports}/{total_imports} passed")
    print(f"Recovery Logic: {'PASS' if test_recovery_logic()[0] else 'FAIL'}")
    print(f"Korean Fonts: {'PASS' if test_korean_fonts()[0] else 'FAIL'}")

    if passed_imports == total_imports and test_recovery_logic()[0]:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
        print("   The BAC calculator applications are ready for use.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Please review the errors above.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
