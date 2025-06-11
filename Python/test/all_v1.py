import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------------------------------
# 1. 파라미터 설정 (필요에 따라 수정)
m = 70.0               # 체중 (kg)
gender = 'male'        # 'male' 또는 'female'
r = 0.68 if gender == 'male' else 0.55  # TBW 분포 비율
V = 500.0              # 음주량 (mL)
ABV = 40.0             # 알코올 도수 (%)
rho_EtOH = 0.789       # 에탄올 밀도 (g/mL)

k1 = 0.2               # 흡수 속도 상수
k2 = 0.1               # 제거 속도 상수
alpha = 0.8            # 흡수 분수 미분 차수 (0 < α ≤ 1)
beta = 0.9             # 제거 분수 미분 차수 (0 < β ≤ 1)

BAChigh = 0.0008       # 0.08% (0.0008 단위)
BAClow = 0.0001        # 0.01% (0.0001 단위)

Tmax = 8.0             # 총 시뮬레이션 시간 (h)
J = 800                # 구간 개수

# ---------------------------------------------------
# 2. 초기 알코올 농도 A0 계산
A0 = (V * (ABV / 100.0) * rho_EtOH) / (r * m)

# 시간 그리드
t = np.linspace(0, Tmax, J+1)  # 0 ~ Tmax, 총 J+1개 포인트

# ---------------------------------------------------
# 3. Mittag–Leffler 함수 근사 (1-파라미터 및 2-파라미터)
def mittag_leffler(z, alpha, max_terms=50, tol=1e-10):
    """
    1-파라미터 Mittage–Leffler 함수 근사:
      E_alpha(z) = sum_{n=0..∞} z^n / Γ(alpha*n + 1)
    """
    result = 0.0
    for n in range(max_terms):
        term = z**n / sp.gamma(alpha*n + 1)
        result += term
        if abs(term) < tol:
            break
    return result

def mittag_leffler_double(x, y, alpha, beta, max_m=20, max_n=20, tol=1e-10):
    """
    2-파라미터(이중) Mittage–Leffler 함수 근사:
      E^{(2)}_{α,β}(x, y) = Σ_{m=0..∞} Σ_{n=0..∞} [ x^m * y^n / Γ(α m + β n + 1) ]
    """
    total = 0.0
    for m_idx in range(max_m):
        for n_idx in range(max_n):
            exponent = alpha*m_idx + beta*n_idx
            denom = sp.gamma(exponent + 1)
            term = (x**m_idx) * (y**n_idx) / denom
            total += term
            if abs(term) < tol:
                break
        # m_idx 증가에 따른 주항이 충분히 작아지면 더이상 계산하지 않음
        if abs((x**m_idx) / sp.gamma(alpha*m_idx + 1)) < tol:
            break
    return total

# ---------------------------------------------------
# 4. 분수 미분 모델에 따른 A(t), B(t) 함수 정의
def A_frac(t):
    """
    A(t) = A0 * E_alpha(-k1 * t^alpha)
    """
    return A0 * np.vectorize(lambda tau: mittag_leffler(-k1 * (tau**alpha), alpha))(t)

def B_frac(t):
    """
    B(t) = k1 * A0 * t^{β - 1} * E^{(2)}_{α, β}(-k1 t^α, -k2 t^β)
    """
    B_vals = np.zeros_like(t)
    for idx, tau in enumerate(t):
        if tau == 0:
            B_vals[idx] = 0.0
        else:
            x = -k1 * (tau**alpha)
            y = -k2 * (tau**beta)
            E2 = mittag_leffler_double(x, y, alpha, beta)
            B_vals[idx] = k1 * A0 * (tau**(beta - 1)) * E2
    return B_vals

# ---------------------------------------------------
# 5. 시간 그리드에서 A, B 계산
A_vals = A_frac(t)
B_vals = B_frac(t)

# ---------------------------------------------------
# 6. 기준(BAC = 0.08%, 0.01%)에 도달하는 시간 찾기 (이분법 사용)
def find_threshold_time(func, threshold, t_start, t_end, tol=1e-5, max_iter=100):
    """
    이분법으로 주어진 func(t) - threshold = 0인 t를 [t_start, t_end] 구간에서 찾음
    """
    a, b = t_start, t_end
    fa = func(a) - threshold
    fb = func(b) - threshold
    if fa * fb > 0:
        return None  # 구간 내에서 부호 변화가 없으면 None

    for _ in range(max_iter):
        mid = (a + b) / 2
        fmid = func(mid) - threshold
        if abs(fmid) < tol:
            return mid
        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
    return (a + b) / 2

# B(t)를 보간(interpolation)해서 연속 함수처럼 사용
B_interp = interp1d(t, B_vals, kind='cubic', fill_value="extrapolate")

# 피크 시점 t_max 찾기
t_max_idx = np.argmax(B_vals)
t_max = t[t_max_idx]

# ti (BAC=0.08%)는 [0, t_max] 구간, tf (BAC=0.01%)는 [t_max, Tmax] 구간에서 찾음
ti = find_threshold_time(B_interp, BAChigh, 0, t_max)
tf = find_threshold_time(B_interp, BAClow, t_max, Tmax)

# 허용(tolerance) 시간 ∆T 계산
delta_T = None
if ti is not None and tf is not None:
    delta_T = tf - ti

# ---------------------------------------------------
# 7. 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(t, B_vals, label='B(t) - BAC Curve')
if ti is not None:
    plt.axvline(x=ti, color='r', linestyle='--', label=f'ti (0.08%) = {ti:.2f} h')
if tf is not None:
    plt.axvline(x=tf, color='g', linestyle='--', label=f'tf (0.01%) = {tf:.2f} h')
plt.axvline(x=t_max, color='b', linestyle='--', label=f'tmax = {t_max:.2f} h')
plt.xlabel('Time (h)')
plt.ylabel('BAC')
plt.title('Fractional-Order BAC Model')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 8. 결과 출력
print(f"Initial A0: {A0:.4f}")
if ti is not None:
    print(f"Time of intoxication threshold (ti): {ti:.4f} h")
else:
    print("ti not found")
if tf is not None:
    print(f"Time of recovery threshold (tf): {tf:.4f} h")
else:
    print("tf not found")
if delta_T is not None:
    print(f"Tolerance time (∆T): {delta_T:.4f} h")
else:
    print("Tolerance time ∆T could not be computed (missing ti or tf).")
