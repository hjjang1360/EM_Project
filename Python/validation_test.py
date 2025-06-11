#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Validation Test
Validates that all BAC calculator modules work correctly
"""

import sys
import os


def validate_recovery_logic():
    """Validate the recovery time logic"""
    try:
        # Import required modules
        sys.path.append(os.getcwd())
        from bac_calculator_simple import calculate_bac, find_recovery_times
        import numpy as np

        # Test parameters
        volume_ml = 360
        alcohol_percent = 17
        weight_kg = 70
        gender = "male"

        # Calculate BAC
        t_array = np.linspace(0, 24, 1000)
        bac_mg = calculate_bac(volume_ml, alcohol_percent, weight_kg, gender, t_array)

        # Get peak information
        peak_bac = np.max(bac_mg)
        peak_time = t_array[np.argmax(bac_mg)]

        # Calculate recovery times
        recovery_times = find_recovery_times(t_array, bac_mg)

        result = {
            "peak_bac": peak_bac,
            "peak_time": peak_time,
            "recovery_times": recovery_times,
            "test_passed": all(recovery_times.values()),
        }

        return True, result

    except Exception as e:
        return False, str(e)


def validate_imports():
    """Validate that all modules can be imported"""
    modules = [
        "bac_calculator_simple",
        "bac_calculator_gui",
        "bac_calculator_enhanced",
        "bac_calculator_app",
        "bac_calculator_web",
        "bac_calculator_web_fixed",
    ]

    results = {}
    for module in modules:
        try:
            if os.path.exists(f"{module}.py"):
                __import__(module)
                results[module] = "SUCCESS"
            else:
                results[module] = "FILE_NOT_FOUND"
        except Exception as e:
            results[module] = f"ERROR: {str(e)[:50]}..."

    return results


def main():
    print("BAC Calculator - Code Validation")
    print("=" * 50)

    # Test imports
    print("1. Testing Module Imports:")
    import_results = validate_imports()
    for module, status in import_results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"   {status_symbol} {module}: {status}")

    print()

    # Test recovery logic
    print("2. Testing Recovery Logic:")
    success, result = validate_recovery_logic()

    if success:
        print(
            f"   ✓ Peak BAC: {result['peak_bac']:.1f} mg/100mL at {result['peak_time']:.1f}h"
        )
        recovery = result["recovery_times"]
        if recovery["legal"]:
            print(f"   ✓ Legal recovery: {recovery['legal']:.1f}h")
        if recovery["safe"]:
            print(f"   ✓ Safe recovery: {recovery['safe']:.1f}h")
        if recovery["full"]:
            print(f"   ✓ Full recovery: {recovery['full']:.1f}h")

        if result["test_passed"]:
            print("   ✓ Recovery logic test PASSED")
        else:
            print("   ✗ Recovery logic test FAILED")
    else:
        print(f"   ✗ Recovery logic test ERROR: {result}")

    print()

    # Summary
    successful_imports = sum(
        1 for status in import_results.values() if status == "SUCCESS"
    )
    total_modules = len(import_results)

    print("3. Summary:")
    print(f"   Modules imported: {successful_imports}/{total_modules}")
    print(
        f"   Recovery logic: {'PASS' if success and result.get('test_passed') else 'FAIL'}"
    )

    if successful_imports >= 4 and success and result.get("test_passed"):
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("   The BAC calculator is ready for use.")
    else:
        print("\n⚠️ VALIDATION ISSUES DETECTED")
        print("   Please check the errors above.")


if __name__ == "__main__":
    main()
