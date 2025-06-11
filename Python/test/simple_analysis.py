"""
Simple demonstration of fractional BAC model issues
"""

import numpy as np

print("=== FRACTIONAL BAC MODEL 문제점 분석 ===")
print()

# 1. 수학적 공식의 불일치
print("1. 수학적 공식 불일치:")
print("   이론적 기반 (PDF에서):")
print("   A(t) = A₀ E_α(-k₁ t^α)")
print("   B(t) = (A₀ k₁)/(k₂ - k₁) [E_α(-k₁ t^α) - E_β(-k₂ t^β)]")
print()
print("   하지만 구현에서는:")
print("   B(t) = k₁ A₀ t^(β-1) E_{α,β}(-k₁t^α, -k₂t^β)  ← 잘못된 공식")
print()

# 2. 매개변수 불일치
print("2. 매개변수 불일치:")
print("   all_v7.py: k1=1.0, k2=0.12")
print("   all_v9.py: k1=1.0, k2=1.2")
print("   LaTeX 문서: k1=0.8, k2=1.0 (권장)")
print()

# 3. 수치적 불안정성
print("3. Mittag-Leffler 함수 구현 문제:")
print("   - overflow 발생")
print("   - 음수 인수 처리 부정확")
print("   - 수렴성 검사 부족")
print()

# 4. 물리적 현실성 부족
print("4. 물리적 현실성 문제:")
print("   - BAC가 피크 후 단조감소하지 않음")
print("   - 비현실적인 피크 값")
print("   - 회복 시간이 생리학적 기대와 맞지 않음")
print()

# 5. 단위 변환 혼란
print("5. 단위 변환 문제:")
print("   - 어떤 곳: B_t * 0.1  (g/L → g/100mL)")
print("   - 다른 곳: B_t * 100  (g/L → mg/100mL)")
print("   - 일관성 없는 적용")
print()

print("=== 주요 해결 방안 ===")
print()
print("1. 이론적으로 올바른 공식 사용:")
print("   B(t) = (A₀ k₁)/(k₂ - k₁) [E_α(-k₁ t^α) - E_β(-k₂ t^β)]")
print()
print("2. 수치적으로 안정한 Mittag-Leffler 함수 구현")
print()
print("3. 일관된 매개변수: k1=0.8, k2=1.0, α=0.8, β=0.9")
print()
print("4. 물리적 제약 조건 적용: 0 ≤ B(t) ≤ A₀")
print()
print("5. 일관된 단위 처리")
print()

print("현재 코드의 가장 큰 문제는 잘못된 수학적 공식을 사용하는 것입니다.")
print("PDF의 이론적 기반과 일치하지 않는 공식들이 여러 버전에서 사용되고 있습니다.")
