"""
BAC Calculator - Simple Command Line Version
사용자의 개인정보와 음주 정보를 입력받아 혈중알코올농도를 계산하고
언제 술이 깨는지 예측하는 간단한 커맨드라인 프로그램
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.special import gamma


class SimpleBACCalculator:
    def __init__(self):
        # Model parameters
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

    def classical_bac_model(self, t, A0, k1, k2):
        """Classical two-compartment BAC model"""
        if t == 0:
            return A0, 0.0

        A_t = A0 * np.exp(-k1 * t)

        if abs(k1 - k2) < 1e-10:
            B_t = k1 * A0 * t * np.exp(-k1 * t)
        else:
            B_t = (k1 * A0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))

        return max(0.0, A_t), max(0.0, B_t)

    def calculate_initial_concentration(self, weight, tbw_ratio, volume, abv):
        """Calculate initial alcohol concentration A0 in stomach"""
        rho_ethanol = 0.789  # g/mL
        alcohol_mass = volume * (abv / 100) * rho_ethanol  # grams
        tbw_volume = tbw_ratio * weight  # liters
        A0 = alcohol_mass / tbw_volume  # g/L        return A0

    def find_recovery_times(self, t_array, bac_array):
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
        recovery_time = (
            post_peak_times[recovery_idx[0]] if len(recovery_idx) > 0 else None
        )

        return legal_time, safe_time, recovery_time

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

    def get_user_input(self):
        """Get user input for calculation"""
        print("🍺 혈중알코올농도(BAC) 계산기")
        print("=" * 40)

        # Personal information
        print("\n👤 개인정보를 입력해주세요:")

        while True:
            try:
                gender = input("성별 (남성/여성): ").strip()
                if gender in ["남성", "여성"]:
                    break
                print("올바른 성별을 입력해주세요 (남성/여성)")
            except:
                print("올바른 값을 입력해주세요")

        while True:
            try:
                age = int(input("나이: "))
                if 19 <= age <= 100:
                    break
                print("올바른 나이를 입력해주세요 (19-100)")
            except:
                print("숫자를 입력해주세요")

        while True:
            try:
                weight = float(input("몸무게 (kg): "))
                if 30 <= weight <= 200:
                    break
                print("올바른 몸무게를 입력해주세요 (30-200kg)")
            except:
                print("숫자를 입력해주세요")

        # Drinking information
        print("\n🍻 음주정보를 입력해주세요:")

        print("술 종류를 선택하세요:")
        print("1. 맥주 (5%, 500mL)")
        print("2. 소주 (17%, 360mL)")
        print("3. 와인 (12%, 150mL)")
        print("4. 위스키 (40%, 50mL)")
        print("5. 막걸리 (6%, 300mL)")
        print("6. 직접입력")

        drink_types = {
            1: {"name": "맥주", "abv": 5, "volume": 500},
            2: {"name": "소주", "abv": 17, "volume": 360},
            3: {"name": "와인", "abv": 12, "volume": 150},
            4: {"name": "위스키", "abv": 40, "volume": 50},
            5: {"name": "막걸리", "abv": 6, "volume": 300},
            6: {"name": "직접입력", "abv": 0, "volume": 0},
        }

        while True:
            try:
                choice = int(input("선택 (1-6): "))
                if choice in drink_types:
                    break
                print("1-6 중에서 선택해주세요")
            except:
                print("숫자를 입력해주세요")

        if choice == 6:
            while True:
                try:
                    volume = float(input("음주량 (mL): "))
                    if 10 <= volume <= 2000:
                        break
                    print("올바른 음주량을 입력해주세요 (10-2000mL)")
                except:
                    print("숫자를 입력해주세요")

            while True:
                try:
                    abv = float(input("알코올 도수 (%): "))
                    if 0.5 <= abv <= 60:
                        break
                    print("올바른 알코올 도수를 입력해주세요 (0.5-60%)")
                except:
                    print("숫자를 입력해주세요")
        else:
            drink_info = drink_types[choice]
            print(f"\n선택된 술: {drink_info['name']}")

            # Allow volume adjustment
            default_volume = drink_info["volume"]
            volume_input = input(
                f"음주량 (기본값: {default_volume}mL, Enter로 기본값 사용): "
            ).strip()

            if volume_input:
                try:
                    volume = float(volume_input)
                    if volume <= 0:
                        volume = default_volume
                except:
                    volume = default_volume
            else:
                volume = default_volume

            abv = drink_info["abv"]

        # Model selection
        print("\n모델을 선택하세요:")
        print("1. 분수계 모델 (메모리 효과 포함, 더 정확)")
        print("2. 고전 모델 (단순한 지수 감소)")

        while True:
            try:
                model_choice = int(input("선택 (1-2): "))
                if model_choice in [1, 2]:
                    break
                print("1 또는 2를 선택해주세요")
            except:
                print("숫자를 입력해주세요")

        model_type = "분수계" if model_choice == 1 else "고전"

        # Drinking time
        drinking_time = input(
            f"음주 시작 시간 (HH:MM, 기본값: {datetime.now().strftime('%H:%M')}): "
        ).strip()
        if not drinking_time:
            drinking_time = datetime.now().strftime("%H:%M")

        return {
            "gender": gender,
            "age": age,
            "weight": weight,
            "volume": volume,
            "abv": abv,
            "model_type": model_type,
            "drinking_time": drinking_time,
        }

    def calculate_and_display(self, user_data):
        """Calculate BAC and display results"""
        # Extract data
        gender = user_data["gender"]
        age = user_data["age"]
        weight = user_data["weight"]
        volume = user_data["volume"]
        abv = user_data["abv"]
        model_type = user_data["model_type"]
        drinking_time = user_data["drinking_time"]

        # Calculate TBW ratio
        if gender == "남성":
            tbw_ratio = 0.68 - (age - 25) * 0.001
        else:
            tbw_ratio = 0.55 - (age - 25) * 0.001

        # Calculate initial concentration
        A0 = self.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

        # Time array (0 to 24 hours)
        t_hours = np.linspace(0, 24, 1000)

        # Calculate BAC over time
        bac_values = []

        for t in t_hours:
            if model_type == "분수계":
                _, B_t = self.fractional_bac_model_corrected(
                    t, A0, self.k1, self.k2, self.alpha, self.beta
                )
            else:
                _, B_t = self.classical_bac_model(t, A0, self.k1, self.k2)

            bac_values.append(B_t)

        bac_array = np.array(bac_values)

        # Find recovery times
        legal_time, safe_time, recovery_time = self.find_recovery_times(
            t_hours, bac_array
        )

        # Calculate statistics
        peak_bac = np.max(bac_array) * 100
        peak_time = t_hours[np.argmax(bac_array)]

        # Display results
        print("\n" + "=" * 60)
        print("🧮 BAC 계산 결과")
        print("=" * 60)

        print(f"\n📋 입력 정보:")
        print(f"• 성별: {gender}")
        print(f"• 나이: {age}세")
        print(f"• 몸무게: {weight}kg")
        print(f"• 체수분 비율: {tbw_ratio:.2f}")
        print(f"• 음주량: {volume}mL")
        print(f"• 알코올 도수: {abv}%")
        print(f"• 순수 알코올: {volume * abv / 100 * 0.789:.1f}g")
        print(f"• 사용 모델: {model_type} 모델")

        print(f"\n📊 계산 결과:")
        print(f"• 초기 농도 (A0): {A0:.3f} g/L")
        print(f"• 최고 BAC: {peak_bac:.1f} mg/100mL")
        print(f"• 최고점 도달: 음주 후 {peak_time:.1f}시간")

        # Parse drinking time
        try:
            current_time = datetime.strptime(drinking_time, "%H:%M").time()
            current_datetime = datetime.combine(datetime.today(), current_time)
            peak_datetime = current_datetime + timedelta(hours=peak_time)
            print(f"• 최고점 예상 시간: {peak_datetime.strftime('%H:%M')}")
        except:
            print("• 시간 계산 오류")

        print(f"\n⏰ 회복 시간 예측:")

        try:
            current_datetime = datetime.combine(
                datetime.today(), datetime.strptime(drinking_time, "%H:%M").time()
            )

            if legal_time:
                legal_datetime = current_datetime + timedelta(hours=legal_time)
                print(
                    f"🚗 운전 가능 (50mg/100mL 이하): 음주 후 {legal_time:.1f}시간 ({legal_datetime.strftime('%H:%M')})"
                )
            else:
                print("🚗 운전 가능: 24시간 내 불가능")

            if safe_time:
                safe_datetime = current_datetime + timedelta(hours=safe_time)
                print(
                    f"✅ 안전 운전 (30mg/100mL 이하): 음주 후 {safe_time:.1f}시간 ({safe_datetime.strftime('%H:%M')})"
                )
            else:
                print("✅ 안전 운전: 24시간 내 불가능")

            if recovery_time:
                recovery_datetime = current_datetime + timedelta(hours=recovery_time)
                print(
                    f"🎉 완전 회복 (10mg/100mL 이하): 음주 후 {recovery_time:.1f}시간 ({recovery_datetime.strftime('%H:%M')})"
                )
            else:
                print("🎉 완전 회복: 24시간 내 불가능")

        except:
            print("시간 계산에 오류가 있습니다.")

        print(f"\n⚠️ 주의사항:")
        print("• 이 계산기는 참고용이며 개인차가 있을 수 있습니다")
        print("• 실제 음주운전은 절대 금지입니다")
        print("• 안전을 위해 음주 후에는 대중교통을 이용하세요")

        # Ask if user wants to see graph
        show_graph = input("\n그래프를 보시겠습니까? (y/n): ").strip().lower()
        if show_graph in ["y", "yes", "예", "ㅇ"]:
            self.show_graph(
                t_hours, bac_array, legal_time, safe_time, recovery_time, model_type
            )

    def show_graph(
        self, t_hours, bac_array, legal_time, safe_time, recovery_time, model_type
    ):
        """Display BAC graph using matplotlib"""
        plt.figure(figsize=(12, 8))

        # Plot BAC curve
        bac_mg = bac_array * 100  # Convert to mg/100mL
        plt.plot(t_hours, bac_mg, "r-", linewidth=3, label="혈중알코올농도")

        # Add threshold lines
        plt.axhline(
            y=80,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="음주운전 단속기준 (80 mg/100mL)",
        )
        plt.axhline(
            y=50,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="면허정지 기준 (50 mg/100mL)",
        )
        plt.axhline(
            y=30,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="안전운전 기준 (30 mg/100mL)",
        )
        plt.axhline(
            y=10,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label="완전회복 기준 (10 mg/100mL)",
        )

        # Add recovery time markers
        if legal_time:
            plt.axvline(
                x=legal_time,
                color="orange",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label=f"운전가능: {legal_time:.1f}h",
            )

        if safe_time:
            plt.axvline(
                x=safe_time,
                color="green",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label=f"안전운전: {safe_time:.1f}h",
            )

        if recovery_time:
            plt.axvline(
                x=recovery_time,
                color="blue",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label=f"완전회복: {recovery_time:.1f}h",
            )

        plt.xlabel("시간 (hours)", fontsize=12)
        plt.ylabel("혈중알코올농도 (mg/100mL)", fontsize=12)
        plt.title(
            f"혈중알코올농도 예측 - {model_type} 모델", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        plt.ylim(0, max(100, np.max(bac_mg) * 1.1))

        # Improve layout
        plt.tight_layout()

        # Show plot
        plt.show()

    def run(self):
        """Main application loop"""
        while True:
            try:
                # Get user input
                user_data = self.get_user_input()

                # Calculate and display results
                self.calculate_and_display(user_data)

                # Ask if user wants to calculate again
                again = input("\n다시 계산하시겠습니까? (y/n): ").strip().lower()
                if again not in ["y", "yes", "예", "ㅇ"]:
                    break

                print("\n" + "=" * 60)

            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                print("다시 시도해주세요.")


def main():
    calculator = SimpleBACCalculator()
    calculator.run()


if __name__ == "__main__":
    main()
