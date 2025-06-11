"""
BAC Calculator - Simple GUI Application
사용자의 개인정보와 음주 정보를 입력받아 혈중알코올농도를 계산하고
언제 술이 깨는지 예측하는 GUI 애플리케이션
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scipy.special import gamma
import matplotlib.dates as mdates


class BACCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍺 혈중알코올농도(BAC) 계산기")
        self.root.geometry("1200x800")

        # Model parameters
        self.k1, self.k2 = 0.8, 1.0
        self.alpha, self.beta = 0.8, 0.9

        self.setup_ui()

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
        A0 = alcohol_mass / tbw_volume  # g/L
        return A0

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

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="입력 정보", padding="10")
        input_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        # Personal information
        ttk.Label(input_frame, text="개인정보", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 10)
        )

        ttk.Label(input_frame, text="성별:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.gender_var = tk.StringVar(value="남성")
        gender_combo = ttk.Combobox(
            input_frame,
            textvariable=self.gender_var,
            values=["남성", "여성"],
            state="readonly",
            width=10,
        )
        gender_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 20))

        ttk.Label(input_frame, text="나이:").grid(
            row=1, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.age_var = tk.StringVar(value="25")
        age_entry = ttk.Entry(input_frame, textvariable=self.age_var, width=10)
        age_entry.grid(row=1, column=3, sticky=tk.W)

        ttk.Label(input_frame, text="몸무게 (kg):").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.weight_var = tk.StringVar(value="70")
        weight_entry = ttk.Entry(input_frame, textvariable=self.weight_var, width=10)
        weight_entry.grid(row=2, column=1, sticky=tk.W, padx=(0, 20))

        ttk.Label(input_frame, text="키 (cm):").grid(
            row=2, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.height_var = tk.StringVar(value="170")
        height_entry = ttk.Entry(input_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=2, column=3, sticky=tk.W)

        # Drinking information
        ttk.Label(input_frame, text="음주정보", font=("Arial", 12, "bold")).grid(
            row=3, column=0, columnspan=4, sticky=tk.W, pady=(20, 10)
        )

        ttk.Label(input_frame, text="술 종류:").grid(
            row=4, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.drink_type_var = tk.StringVar(value="맥주")
        drink_combo = ttk.Combobox(
            input_frame,
            textvariable=self.drink_type_var,
            values=["맥주", "소주", "와인", "위스키", "막걸리", "직접입력"],
            state="readonly",
            width=15,
        )
        drink_combo.grid(row=4, column=1, sticky=tk.W, padx=(0, 20))
        drink_combo.bind("<<ComboboxSelected>>", self.on_drink_type_change)

        ttk.Label(input_frame, text="음주량 (mL):").grid(
            row=4, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.volume_var = tk.StringVar(value="500")
        volume_entry = ttk.Entry(input_frame, textvariable=self.volume_var, width=10)
        volume_entry.grid(row=4, column=3, sticky=tk.W)

        ttk.Label(input_frame, text="알코올 도수 (%):").grid(
            row=5, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.abv_var = tk.StringVar(value="5")
        self.abv_entry = ttk.Entry(input_frame, textvariable=self.abv_var, width=10)
        self.abv_entry.grid(row=5, column=1, sticky=tk.W, padx=(0, 20))

        ttk.Label(input_frame, text="음주 시작 시간:").grid(
            row=5, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.drinking_time_var = tk.StringVar(value=datetime.now().strftime("%H:%M"))
        time_entry = ttk.Entry(
            input_frame, textvariable=self.drinking_time_var, width=10
        )
        time_entry.grid(row=5, column=3, sticky=tk.W)

        # Model selection
        ttk.Label(input_frame, text="모델:").grid(
            row=6, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0)
        )
        self.model_var = tk.StringVar(value="분수계 모델")
        model_combo = ttk.Combobox(
            input_frame,
            textvariable=self.model_var,
            values=["분수계 모델", "고전 모델"],
            state="readonly",
            width=15,
        )
        model_combo.grid(row=6, column=1, sticky=tk.W, padx=(0, 20), pady=(10, 0))

        # Calculate button
        calc_button = ttk.Button(
            input_frame, text="🧮 BAC 계산하기", command=self.calculate_bac
        )
        calc_button.grid(row=6, column=2, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="결과", padding="10")
        results_frame.grid(
            row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10)
        )
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results text
        self.results_text = tk.Text(
            results_frame, width=40, height=20, font=("Consolas", 10)
        )
        scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="BAC 그래프", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Initialize with empty plot
        self.ax.set_xlabel("시간 (hours)")
        self.ax.set_ylabel("BAC (mg/100mL)")
        self.ax.set_title("혈중알코올농도 예측")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def on_drink_type_change(self, event=None):
        """Update alcohol content and volume based on drink type"""
        drink_types = {
            "맥주": {"abv": 5, "volume": 500},
            "소주": {"abv": 17, "volume": 360},
            "와인": {"abv": 12, "volume": 150},
            "위스키": {"abv": 40, "volume": 50},
            "막걸리": {"abv": 6, "volume": 300},
            "직접입력": {"abv": 20, "volume": 200},
        }

        drink_type = self.drink_type_var.get()
        if drink_type in drink_types:
            self.abv_var.set(str(drink_types[drink_type]["abv"]))
            self.volume_var.set(str(drink_types[drink_type]["volume"]))

            # Enable/disable ABV entry for custom input
            if drink_type == "직접입력":
                self.abv_entry.config(state="normal")
            else:
                self.abv_entry.config(state="readonly")

    def calculate_bac(self):
        try:
            # Get input values
            gender = self.gender_var.get()
            age = int(self.age_var.get())
            weight = float(self.weight_var.get())
            height = float(self.height_var.get())
            volume = float(self.volume_var.get())
            abv = float(self.abv_var.get())
            drinking_time = self.drinking_time_var.get()
            model_type = self.model_var.get()

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
                if model_type == "분수계 모델":
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

            # Update plot
            self.update_plot(t_hours, bac_array, legal_time, safe_time, recovery_time)

            # Update results text
            self.update_results(
                A0,
                bac_array,
                t_hours,
                legal_time,
                safe_time,
                recovery_time,
                drinking_time,
                gender,
                age,
                weight,
                volume,
                abv,
                model_type,
            )

        except ValueError as e:
            messagebox.showerror("입력 오류", f"올바른 숫자를 입력해주세요: {e}")
        except Exception as e:
            messagebox.showerror("계산 오류", f"계산 중 오류가 발생했습니다: {e}")

    def update_plot(self, t_hours, bac_array, legal_time, safe_time, recovery_time):
        """Update the BAC plot"""
        self.ax.clear()

        # Plot BAC curve
        bac_mg = bac_array * 100  # Convert to mg/100mL
        self.ax.plot(t_hours, bac_mg, "r-", linewidth=2, label="혈중알코올농도")

        # Add threshold lines
        self.ax.axhline(
            y=80, color="red", linestyle="--", alpha=0.7, label="음주운전 단속기준 (80)"
        )
        self.ax.axhline(
            y=50, color="orange", linestyle="--", alpha=0.7, label="면허정지 기준 (50)"
        )
        self.ax.axhline(
            y=30, color="green", linestyle="--", alpha=0.7, label="안전운전 기준 (30)"
        )
        self.ax.axhline(
            y=10, color="blue", linestyle="--", alpha=0.7, label="완전회복 기준 (10)"
        )

        # Add recovery time markers
        if legal_time:
            self.ax.axvline(
                x=legal_time,
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f"운전가능: {legal_time:.1f}h",
            )

        if safe_time:
            self.ax.axvline(
                x=safe_time,
                color="green",
                linestyle=":",
                alpha=0.8,
                label=f"안전운전: {safe_time:.1f}h",
            )

        if recovery_time:
            self.ax.axvline(
                x=recovery_time,
                color="blue",
                linestyle=":",
                alpha=0.8,
                label=f"완전회복: {recovery_time:.1f}h",
            )

        self.ax.set_xlabel("시간 (hours)")
        self.ax.set_ylabel("BAC (mg/100mL)")
        self.ax.set_title("혈중알코올농도 예측")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 24)
        self.ax.set_ylim(0, max(100, np.max(bac_mg) * 1.1))

        self.canvas.draw()

    def update_results(
        self,
        A0,
        bac_array,
        t_hours,
        legal_time,
        safe_time,
        recovery_time,
        drinking_time,
        gender,
        age,
        weight,
        volume,
        abv,
        model_type,
    ):
        """Update the results text"""
        self.results_text.delete(1.0, tk.END)

        # Calculate statistics
        peak_bac = np.max(bac_array) * 100
        peak_time = t_hours[np.argmax(bac_array)]

        # Parse drinking time
        try:
            current_time = datetime.strptime(drinking_time, "%H:%M").time()
            current_datetime = datetime.combine(datetime.today(), current_time)
        except:
            current_datetime = datetime.now()

        results = f"""=== BAC 계산 결과 ===

📋 입력 정보:
• 성별: {gender}
• 나이: {age}세
• 몸무게: {weight}kg
• 음주량: {volume}mL
• 알코올 도수: {abv}%
• 순수 알코올: {volume * abv / 100 * 0.789:.1f}g
• 모델: {model_type}

📊 계산 결과:
• 초기 농도 (A0): {A0:.3f} g/L
• 최고 BAC: {peak_bac:.1f} mg/100mL
• 최고점 도달: {peak_time:.1f}시간 후
• 예상 시간: {(current_datetime + timedelta(hours=peak_time)).strftime('%H:%M')}

⏰ 회복 시간 예측:
"""

        if legal_time:
            legal_datetime = current_datetime + timedelta(hours=legal_time)
            results += f"🚗 운전 가능 (50mg/100mL 이하):\n   {legal_time:.1f}시간 후 ({legal_datetime.strftime('%H:%M')})\n\n"
        else:
            results += "🚗 운전 가능: 24시간 내 불가\n\n"

        if safe_time:
            safe_datetime = current_datetime + timedelta(hours=safe_time)
            results += f"✅ 안전 운전 (30mg/100mL 이하):\n   {safe_time:.1f}시간 후 ({safe_datetime.strftime('%H:%M')})\n\n"
        else:
            results += "✅ 안전 운전: 24시간 내 불가\n\n"

        if recovery_time:
            recovery_datetime = current_datetime + timedelta(hours=recovery_time)
            results += f"🎉 완전 회복 (10mg/100mL 이하):\n   {recovery_time:.1f}시간 후 ({recovery_datetime.strftime('%H:%M')})\n\n"
        else:
            results += "🎉 완전 회복: 24시간 내 불가\n\n"

        results += """⚠️ 주의사항:
• 이 계산기는 참고용이며 개인차가 있습니다
• 실제 음주운전은 절대 금지입니다
• 안전을 위해 대중교통을 이용하세요
"""

        self.results_text.insert(tk.END, results)


def main():
    root = tk.Tk()
    app = BACCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
