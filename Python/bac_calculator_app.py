"""
BAC Calculator Application
사용자의 개인정보와 음주 정보를 입력받아 혈중알코올농도를 계산하고
언제 술이 깨는지 예측하는 웹 애플리케이션
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from scipy.special import gamma

# Set page config
st.set_page_config(page_title="🍺 BAC Calculator", page_icon="🍺", layout="wide")

# Title and description
st.title("🍺 혈중알코올농도(BAC) 계산기")
st.markdown("### 언제 술이 깰까요? 🤔")
st.markdown("개인정보와 음주량을 입력하면 혈중알코올농도와 회복시간을 예측해드립니다!")


# Fractional BAC model functions (from our corrected implementation)
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


def fractional_bac_model_corrected(t, A0, k1, k2, alpha=0.8, beta=0.9):
    """Theoretically correct fractional BAC model"""
    if t == 0:
        return A0, 0.0

    if t < 0:
        return A0, 0.0

    # Stomach concentration A(t)
    A_t = A0 * ml1_stable(-k1 * (t**alpha), alpha)

    # Blood concentration B(t)
    if t > 0:
        if abs(k2 - k1) < 1e-10:
            # Special case: k1 ≈ k2
            B_t = A0 * k1 * (t**alpha) * ml1_stable(-k1 * (t**alpha), alpha)
        else:
            # General case: k1 ≠ k2
            term1 = ml1_stable(-k1 * (t**alpha), alpha)
            term2 = ml1_stable(-k2 * (t**beta), beta)
            B_t = (A0 * k1 / (k2 - k1)) * (term1 - term2)

        # Ensure physical constraints
        B_t = max(0.0, min(B_t, A0))
    else:
        B_t = 0.0

    return max(0.0, A_t), B_t


def classical_bac_model(t, A0, k1, k2):
    """Classical two-compartment BAC model"""
    if t == 0:
        return A0, 0.0

    A_t = A0 * np.exp(-k1 * t)

    if abs(k1 - k2) < 1e-10:
        B_t = k1 * A0 * t * np.exp(-k1 * t)
    else:
        B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))

    return max(0.0, A_t), max(0.0, B_t)


def calculate_initial_concentration(weight, tbw_ratio, volume, abv):
    """Calculate initial alcohol concentration A0 in stomach"""
    rho_ethanol = 0.789  # g/mL
    alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
    tbw_volume = tbw_ratio * weight  # liters
    A0 = alcohol_mass / tbw_volume  # g/L
    return A0


def find_recovery_times(t_array, bac_array):
    """Find recovery times for different thresholds - FIXED VERSION"""
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
    recovery_time = post_peak_times[recovery_idx[0]] if len(recovery_idx) > 0 else None

    return legal_time, safe_time, recovery_time


# Sidebar for user input
st.sidebar.header("👤 개인정보")

# Personal information
gender = st.sidebar.selectbox("성별", ["남성", "여성"])
age = st.sidebar.slider("나이", 19, 80, 25)
weight = st.sidebar.slider("몸무게 (kg)", 40, 120, 70)
height = st.sidebar.slider("키 (cm)", 140, 200, 170)

# Calculate TBW ratio based on gender and other factors
if gender == "남성":
    tbw_ratio = 0.68 - (age - 25) * 0.001  # Slightly decrease with age
else:
    tbw_ratio = 0.55 - (age - 25) * 0.001

st.sidebar.markdown(f"**체수분 비율: {tbw_ratio:.2f}**")

st.sidebar.header("🍻 음주정보")

# Alcohol information
drink_types = {
    "맥주": {"abv": 5, "typical_volume": 500},
    "소주": {"abv": 17, "typical_volume": 360},
    "와인": {"abv": 12, "typical_volume": 150},
    "위스키": {"abv": 40, "typical_volume": 50},
    "막걸리": {"abv": 6, "typical_volume": 300},
    "직접입력": {"abv": 20, "typical_volume": 200},
}

drink_type = st.sidebar.selectbox("술 종류", list(drink_types.keys()))

if drink_type == "직접입력":
    abv = st.sidebar.slider("알코올 도수 (%)", 0.5, 60.0, 20.0, 0.5)
    volume = st.sidebar.slider("음주량 (mL)", 10, 1000, 200)
else:
    abv = drink_types[drink_type]["abv"]
    volume = st.sidebar.slider(
        f"{drink_type} 음주량 (mL)", 50, 1000, drink_types[drink_type]["typical_volume"]
    )

st.sidebar.markdown(f"**알코올 도수: {abv}%**")

# Drinking time
drinking_time = st.sidebar.time_input("음주 시작 시간", datetime.now().time())

# Model selection
model_type = st.sidebar.selectbox(
    "모델 선택",
    ["분수계 모델 (메모리 효과 포함)", "고전 모델"],
    help="분수계 모델은 메모리 효과를 고려하여 더 현실적인 예측을 제공합니다",
)

# Calculate button
if st.sidebar.button("🧮 BAC 계산하기", type="primary"):
    # Model parameters
    k1, k2 = 0.8, 1.0
    alpha, beta = 0.8, 0.9

    # Calculate initial concentration
    A0 = calculate_initial_concentration(weight, tbw_ratio, volume, abv)

    # Time array (0 to 24 hours)
    t_hours = np.linspace(0, 24, 1000)

    # Calculate BAC over time
    bac_values = []
    stomach_values = []

    for t in t_hours:
        if model_type == "분수계 모델 (메모리 효과 포함)":
            A_t, B_t = fractional_bac_model_corrected(t, A0, k1, k2, alpha, beta)
        else:
            A_t, B_t = classical_bac_model(t, A0, k1, k2)

        stomach_values.append(A_t)
        bac_values.append(B_t)

    bac_array = np.array(bac_values)
    stomach_array = np.array(stomach_values)

    # Find recovery times
    legal_time, safe_time, recovery_time = find_recovery_times(t_hours, bac_array)

    # Create main columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 BAC 예측 그래프")

        # Create interactive plot with Plotly
        fig = go.Figure()

        # BAC curve
        fig.add_trace(
            go.Scatter(
                x=t_hours,
                y=bac_array * 100,  # Convert to mg/100mL
                mode="lines",
                name="혈중알코올농도",
                line=dict(color="red", width=3),
                hovertemplate="시간: %{x:.1f}시간<br>BAC: %{y:.1f} mg/100mL<extra></extra>",
            )
        )

        # Add threshold lines
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="red",
            annotation_text="음주운전 단속기준 (80 mg/100mL)",
        )
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="orange",
            annotation_text="면허정지 기준 (50 mg/100mL)",
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="안전운전 기준 (30 mg/100mL)",
        )

        # Add recovery time markers
        if legal_time:
            fig.add_vline(
                x=legal_time,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"법적기준 도달: {legal_time:.1f}h",
            )

        if safe_time:
            fig.add_vline(
                x=safe_time,
                line_dash="dot",
                line_color="green",
                annotation_text=f"안전운전 가능: {safe_time:.1f}h",
            )

        fig.update_layout(
            title=f"{model_type} - BAC 예측",
            xaxis_title="시간 (hours)",
            yaxis_title="혈중알코올농도 (mg/100mL)",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("⏰ 회복 시간 예측")

        # Create time predictions
        current_time = datetime.combine(datetime.today(), drinking_time)

        # Peak BAC
        peak_bac = np.max(bac_array) * 100
        peak_time = t_hours[np.argmax(bac_array)]
        peak_datetime = current_time + timedelta(hours=peak_time)

        st.metric(
            "최고 BAC", f"{peak_bac:.1f} mg/100mL", f"음주 후 {peak_time:.1f}시간"
        )
        st.write(f"**예상 시간:** {peak_datetime.strftime('%H:%M')}")

        st.markdown("---")

        # Recovery times
        if legal_time:
            legal_datetime = current_time + timedelta(hours=legal_time)
            st.success(f"🚗 **운전 가능 (50mg/100mL 이하)**")
            st.write(f"음주 후 {legal_time:.1f}시간")
            st.write(f"예상 시간: {legal_datetime.strftime('%H:%M')}")
        else:
            st.error("24시간 내 운전 불가")

        if safe_time:
            safe_datetime = current_time + timedelta(hours=safe_time)
            st.success(f"✅ **안전 운전 (30mg/100mL 이하)**")
            st.write(f"음주 후 {safe_time:.1f}시간")
            st.write(f"예상 시간: {safe_datetime.strftime('%H:%M')}")
        else:
            st.error("24시간 내 안전운전 불가")

        if recovery_time:
            recovery_datetime = current_time + timedelta(hours=recovery_time)
            st.success(f"🎉 **완전 회복 (10mg/100mL 이하)**")
            st.write(f"음주 후 {recovery_time:.1f}시간")
            st.write(f"예상 시간: {recovery_datetime.strftime('%H:%M')}")
        else:
            st.error("24시간 내 완전회복 불가")

    # Additional information
    st.header("📋 상세 정보")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.subheader("입력 정보")
        st.write(f"**성별:** {gender}")
        st.write(f"**나이:** {age}세")
        st.write(f"**몸무게:** {weight}kg")
        st.write(f"**체수분 비율:** {tbw_ratio:.2f}")

    with col4:
        st.subheader("음주 정보")
        st.write(f"**술 종류:** {drink_type}")
        st.write(f"**음주량:** {volume}mL")
        st.write(f"**알코올 도수:** {abv}%")
        st.write(f"**순수 알코올:** {volume * abv / 100 * 0.789:.1f}g")

    with col5:
        st.subheader("계산 결과")
        st.write(f"**초기 농도 (A0):** {A0:.3f} g/L")
        st.write(f"**모델:** {model_type}")
        st.write(f"**최고 BAC:** {peak_bac:.1f} mg/100mL")
        st.write(f"**최고점 도달:** {peak_time:.1f}시간 후")

# Warning message
st.sidebar.markdown("---")
st.sidebar.warning(
    """
⚠️ **주의사항**
- 이 계산기는 참고용이며 개인차가 있을 수 있습니다
- 실제 음주운전은 절대 금지입니다
- 안전을 위해 음주 후에는 대중교통을 이용하세요
"""
)

# Information section
st.markdown("---")
st.header("ℹ️ 정보")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader("한국 음주운전 기준")
    st.markdown(
        """
    - **면허정지:** 50-80 mg/100mL
    - **면허취소:** 80 mg/100mL 이상
    - **안전 기준:** 30 mg/100mL 이하 권장
    """
    )

with info_col2:
    st.subheader("모델 설명")
    st.markdown(
        """
    - **고전 모델:** 전통적인 지수 감소 모델
    - **분수계 모델:** 메모리 효과를 고려한 더 정확한 모델
    - 분수계 모델이 더 현실적인 예측을 제공합니다
    """
    )

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ using Streamlit | 건국대학교 공업수학1 프로젝트*")
