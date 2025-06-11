#!/usr/bin/env python3
"""
Enhanced BAC Calculator - Modern GUI Application
사용자의 개인정보와 음주 정보를 입력받아 혈중알코올농도를 계산하고
언제 술이 깨는지 예측하는 현대적인 GUI 애플리케이션
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from scipy.special import gamma
import matplotlib.dates as mdates


class EnhancedBACCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍺 혈중알코올농도(BAC) 계산기 - Enhanced Version")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Model parameters
        self.k1, self.k2 = 0.8, 1.0
        self.alpha, self.beta = 0.8, 0.9

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure custom styles
        self.configure_styles()

        self.setup_ui()

    def configure_styles(self):
        """Configure custom styles for better appearance"""
        # Configure frame styles
        self.style.configure("Title.TFrame", background="#2c3e50")
        self.style.configure(
            "Input.TFrame", background="#ecf0f1", relief="raised", borderwidth=2
        )
        self.style.configure(
            "Result.TFrame", background="#e8f5e8", relief="raised", borderwidth=2
        )

        # Configure label styles
        self.style.configure(
            "Title.TLabel",
            background="#2c3e50",
            foreground="white",
            font=("Arial", 16, "bold"),
        )
        self.style.configure(
            "Heading.TLabel", font=("Arial", 12, "bold"), background="#ecf0f1"
        )
        self.style.configure("Info.TLabel", font=("Arial", 10), background="#ecf0f1")
        self.style.configure("Result.TLabel", font=("Arial", 10), background="#e8f5e8")

        # Configure button styles
        self.style.configure("Calculate.TButton", font=("Arial", 12, "bold"))
        self.style.configure("Reset.TButton", font=("Arial", 10))

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
        """Find recovery times for different thresholds"""
        bac_mg = bac_array * 100  # Convert to mg/100mL

        # Find peak BAC first
        peak_idx = np.argmax(bac_mg)
        peak_time = t_array[peak_idx]

        # Only look for recovery times after the peak
        post_peak_mask = t_array > peak_time
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

    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Title frame
        title_frame = ttk.Frame(main_container, style="Title.TFrame")
        title_frame.pack(fill="x", pady=(0, 15))

        title_label = ttk.Label(
            title_frame, text="🍺 혈중알코올농도(BAC) 계산기", style="Title.TLabel"
        )
        title_label.pack(pady=15)

        subtitle_label = ttk.Label(
            title_frame,
            text="분수계 미분방정식을 이용한 정확한 BAC 예측 시스템",
            style="Title.TLabel",
            font=("Arial", 10),
        )
        subtitle_label.pack(pady=(0, 15))

        # Create notebook for organized input
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill="both", expand=True)

        # Input tab
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="📝 정보 입력")

        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="📊 결과 분석")

        # Setup input tab
        self.setup_input_tab(input_frame)

        # Setup results tab
        self.setup_results_tab(results_frame)

    def setup_input_tab(self, parent):
        """Setup the input tab with improved layout"""
        # Main input container
        input_container = ttk.Frame(parent)
        input_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel for inputs
        left_panel = ttk.Frame(input_container)
        left_panel.pack(side="left", fill="y", padx=(0, 20))

        # Right panel for quick graph
        right_panel = ttk.Frame(input_container)
        right_panel.pack(side="right", fill="both", expand=True)

        # Personal Information Section
        personal_frame = ttk.LabelFrame(left_panel, text="👤 개인정보", padding=15)
        personal_frame.pack(fill="x", pady=(0, 15))

        # Gender
        ttk.Label(personal_frame, text="성별:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.gender_var = tk.StringVar(value="남성")
        gender_combo = ttk.Combobox(
            personal_frame,
            textvariable=self.gender_var,
            values=["남성", "여성"],
            state="readonly",
            width=15,
        )
        gender_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)

        # Age
        ttk.Label(personal_frame, text="나이:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.age_var = tk.IntVar(value=25)
        age_spin = ttk.Spinbox(
            personal_frame, from_=19, to=100, textvariable=self.age_var, width=15
        )
        age_spin.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        # Weight
        ttk.Label(personal_frame, text="몸무게 (kg):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.weight_var = tk.DoubleVar(value=70.0)
        weight_spin = ttk.Spinbox(
            personal_frame,
            from_=30,
            to=200,
            increment=0.5,
            textvariable=self.weight_var,
            width=15,
        )
        weight_spin.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # Height (optional)
        ttk.Label(personal_frame, text="키 (cm, 선택):").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.height_var = tk.DoubleVar(value=170.0)
        height_spin = ttk.Spinbox(
            personal_frame,
            from_=140,
            to=220,
            increment=0.5,
            textvariable=self.height_var,
            width=15,
        )
        height_spin.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=5)

        # Drinking Information Section
        drinking_frame = ttk.LabelFrame(left_panel, text="🍻 음주정보", padding=15)
        drinking_frame.pack(fill="x", pady=(0, 15))

        # Drink type
        ttk.Label(drinking_frame, text="술 종류:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.drink_var = tk.StringVar(value="소주")
        drink_combo = ttk.Combobox(
            drinking_frame,
            textvariable=self.drink_var,
            values=["맥주", "소주", "와인", "위스키", "막걸리", "직접입력"],
            state="readonly",
            width=15,
        )
        drink_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)
        drink_combo.bind("<<ComboboxSelected>>", self.on_drink_change)

        # Volume
        ttk.Label(drinking_frame, text="음주량 (mL):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.volume_var = tk.DoubleVar(value=360.0)
        self.volume_spin = ttk.Spinbox(
            drinking_frame,
            from_=10,
            to=2000,
            increment=10,
            textvariable=self.volume_var,
            width=15,
        )
        self.volume_spin.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        # ABV
        ttk.Label(drinking_frame, text="알코올 도수 (%):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.abv_var = tk.DoubleVar(value=17.0)
        self.abv_spin = ttk.Spinbox(
            drinking_frame,
            from_=0.5,
            to=60,
            increment=0.5,
            textvariable=self.abv_var,
            width=15,
        )
        self.abv_spin.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # Drinking time
        ttk.Label(drinking_frame, text="음주 시작 시간:").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.time_var = tk.StringVar(value=datetime.now().strftime("%H:%M"))
        time_entry = ttk.Entry(drinking_frame, textvariable=self.time_var, width=15)
        time_entry.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=5)

        # Model Selection
        model_frame = ttk.LabelFrame(left_panel, text="🧮 계산 모델", padding=15)
        model_frame.pack(fill="x", pady=(0, 15))

        self.model_var = tk.StringVar(value="분수계")
        ttk.Radiobutton(
            model_frame,
            text="분수계 모델 (정확, 권장)",
            variable=self.model_var,
            value="분수계",
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            model_frame,
            text="고전 모델 (단순, 빠름)",
            variable=self.model_var,
            value="고전",
        ).pack(anchor="w", pady=2)

        # Calculate button
        calc_button = ttk.Button(
            left_panel,
            text="🧮 BAC 계산하기",
            command=self.calculate_bac,
            style="Calculate.TButton",
        )
        calc_button.pack(fill="x", pady=10)

        # Reset button
        reset_button = ttk.Button(
            left_panel,
            text="🔄 초기화",
            command=self.reset_inputs,
            style="Reset.TButton",
        )
        reset_button.pack(fill="x", pady=(0, 10))

        # Quick preview graph in right panel
        preview_frame = ttk.LabelFrame(
            right_panel, text="📈 실시간 미리보기", padding=10
        )
        preview_frame.pack(fill="both", expand=True)

        # Create preview figure
        self.preview_fig = Figure(figsize=(6, 4), dpi=80)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initialize preview
        self.update_preview()

        # Bind events for real-time updates
        for var in [
            self.gender_var,
            self.age_var,
            self.weight_var,
            self.volume_var,
            self.abv_var,
            self.model_var,
        ]:
            var.trace("w", lambda *args: self.update_preview())

    def setup_results_tab(self, parent):
        """Setup the results tab"""
        results_container = ttk.Frame(parent)
        results_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel for text results
        left_results = ttk.Frame(results_container)
        left_results.pack(side="left", fill="y", padx=(0, 20))

        # Results text frame
        text_frame = ttk.LabelFrame(left_results, text="📋 계산 결과", padding=15)
        text_frame.pack(fill="both", expand=True)

        self.results_text = tk.Text(
            text_frame, wrap="word", height=25, width=50, font=("Consolas", 10)
        )

        # Scrollbar for text
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Right panel for main graph
        right_results = ttk.Frame(results_container)
        right_results.pack(side="right", fill="both", expand=True)

        graph_frame = ttk.LabelFrame(
            right_results, text="📊 BAC 변화 그래프", padding=10
        )
        graph_frame.pack(fill="both", expand=True)

        # Create main figure
        self.main_fig = Figure(figsize=(10, 8), dpi=100)
        self.main_ax = self.main_fig.add_subplot(111)
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, graph_frame)
        self.main_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Graph controls
        controls_frame = ttk.Frame(right_results)
        controls_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(controls_frame, text="📊 그래프 저장", command=self.save_graph).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(controls_frame, text="📄 결과 저장", command=self.save_results).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(
            controls_frame, text="🎯 비교 분석", command=self.compare_models
        ).pack(side="left")

    def on_drink_change(self, event=None):
        """Handle drink type change"""
        drink_presets = {
            "맥주": {"abv": 5.0, "volume": 500.0},
            "소주": {"abv": 17.0, "volume": 360.0},
            "와인": {"abv": 12.0, "volume": 150.0},
            "위스키": {"abv": 40.0, "volume": 50.0},
            "막걸리": {"abv": 6.0, "volume": 300.0},
            "직접입력": {"abv": 20.0, "volume": 100.0},
        }

        drink_type = self.drink_var.get()
        if drink_type in drink_presets:
            preset = drink_presets[drink_type]
            self.abv_var.set(preset["abv"])
            self.volume_var.set(preset["volume"])

            # Enable/disable ABV control for custom input
            if drink_type == "직접입력":
                self.abv_spin.configure(state="normal")
            else:
                self.abv_spin.configure(state="readonly")

    def update_preview(self):
        """Update the preview graph"""
        try:
            # Get current values
            gender = self.gender_var.get()
            age = self.age_var.get()
            weight = self.weight_var.get()
            volume = self.volume_var.get()
            abv = self.abv_var.get()
            model_type = self.model_var.get()

            # Calculate TBW ratio
            if gender == "남성":
                tbw_ratio = 0.68 - (age - 25) * 0.001
            else:
                tbw_ratio = 0.55 - (age - 25) * 0.001

            # Calculate initial concentration
            A0 = self.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

            # Time array for preview
            t_hours = np.linspace(0, 12, 100)
            bac_values = []

            for t in t_hours:
                if model_type == "분수계":
                    _, B_t = self.fractional_bac_model_corrected(
                        t, A0, self.k1, self.k2, self.alpha, self.beta
                    )
                else:
                    _, B_t = self.classical_bac_model(t, A0, self.k1, self.k2)

                bac_values.append(B_t * 100)  # Convert to mg/100mL

            # Clear and plot
            self.preview_ax.clear()
            self.preview_ax.plot(
                t_hours, bac_values, "b-", linewidth=2, label="BAC 예측"
            )

            # Add threshold lines
            self.preview_ax.axhline(
                y=80, color="red", linestyle="--", alpha=0.7, label="단속기준"
            )
            self.preview_ax.axhline(
                y=50, color="orange", linestyle="--", alpha=0.7, label="면허정지"
            )
            self.preview_ax.axhline(
                y=30, color="green", linestyle="--", alpha=0.7, label="안전운전"
            )

            self.preview_ax.set_xlabel("시간 (hours)")
            self.preview_ax.set_ylabel("BAC (mg/100mL)")
            self.preview_ax.set_title(f"{model_type} 모델 미리보기")
            self.preview_ax.legend(fontsize=8)
            self.preview_ax.grid(True, alpha=0.3)
            self.preview_ax.set_xlim(0, 12)
            self.preview_ax.set_ylim(
                0, max(100, max(bac_values) * 1.1) if bac_values else 100
            )

            self.preview_fig.tight_layout()
            self.preview_canvas.draw()

        except Exception as e:
            # Handle errors silently in preview
            pass

    def calculate_bac(self):
        """Calculate BAC and display detailed results"""
        try:
            # Get input values
            gender = self.gender_var.get()
            age = self.age_var.get()
            weight = self.weight_var.get()
            volume = self.volume_var.get()
            abv = self.abv_var.get()
            model_type = self.model_var.get()
            drinking_time = self.time_var.get()

            # Calculate TBW ratio
            if gender == "남성":
                tbw_ratio = 0.68 - (age - 25) * 0.001
            else:
                tbw_ratio = 0.55 - (age - 25) * 0.001

            # Calculate initial concentration
            A0 = self.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

            # Time array (0 to 24 hours)
            t_hours = np.linspace(0, 24, 1000)
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
            self.display_results(
                gender,
                age,
                weight,
                volume,
                abv,
                model_type,
                drinking_time,
                tbw_ratio,
                A0,
                peak_bac,
                peak_time,
                legal_time,
                safe_time,
                recovery_time,
            )

            # Plot main graph
            self.plot_main_graph(
                t_hours,
                bac_array,
                legal_time,
                safe_time,
                recovery_time,
                model_type,
                drinking_time,
            )

            # Switch to results tab
            self.root.nametowidget(".!frame.!notebook").select(1)

        except Exception as e:
            messagebox.showerror("계산 오류", f"계산 중 오류가 발생했습니다:\n{str(e)}")

    def display_results(
        self,
        gender,
        age,
        weight,
        volume,
        abv,
        model_type,
        drinking_time,
        tbw_ratio,
        A0,
        peak_bac,
        peak_time,
        legal_time,
        safe_time,
        recovery_time,
    ):
        """Display detailed results in text format"""
        self.results_text.delete(1.0, tk.END)

        # Calculate alcohol mass
        alcohol_mass = volume * abv / 100 * 0.789

        results = f"""
🧮 BAC 계산 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📋 입력 정보:
• 성별: {gender}
• 나이: {age}세
• 몸무게: {weight}kg
• 체수분 비율: {tbw_ratio:.2%}
• 음주량: {volume}mL
• 알코올 도수: {abv}%
• 순수 알코올: {alcohol_mass:.1f}g
• 사용 모델: {model_type} 모델
• 음주 시작: {drinking_time}

📊 계산 결과:
• 초기 농도 (A0): {A0:.3f} g/L
• 최고 BAC: {peak_bac:.1f} mg/100mL
• 최고점 도달: 음주 후 {peak_time:.1f}시간

⏰ 회복 시간 예측:
"""

        try:
            current_time = datetime.strptime(drinking_time, "%H:%M").time()
            current_datetime = datetime.combine(datetime.today(), current_time)
            peak_datetime = current_datetime + timedelta(hours=peak_time)
            results += f"• 최고점 예상 시간: {peak_datetime.strftime('%H:%M')}\n\n"

            if legal_time:
                legal_datetime = current_datetime + timedelta(hours=legal_time)
                results += f"🚗 운전 가능 (50mg/100mL 이하):\n"
                results += f"   음주 후 {legal_time:.1f}시간 ({legal_datetime.strftime('%H:%M')})\n\n"
            else:
                results += "🚗 운전 가능: 24시간 내 불가능\n\n"

            if safe_time:
                safe_datetime = current_datetime + timedelta(hours=safe_time)
                results += f"✅ 안전 운전 (30mg/100mL 이하):\n"
                results += f"   음주 후 {safe_time:.1f}시간 ({safe_datetime.strftime('%H:%M')})\n\n"
            else:
                results += "✅ 안전 운전: 24시간 내 불가능\n\n"

            if recovery_time:
                recovery_datetime = current_datetime + timedelta(hours=recovery_time)
                results += f"🎉 완전 회복 (10mg/100mL 이하):\n"
                results += f"   음주 후 {recovery_time:.1f}시간 ({recovery_datetime.strftime('%H:%M')})\n\n"
            else:
                results += "🎉 완전 회복: 24시간 내 불가능\n\n"

        except:
            results += "시간 계산에 오류가 있습니다.\n\n"

        results += f"""
📊 BAC 단계별 기준:
• 80 mg/100mL 이상: 음주운전 단속 대상
• 50 mg/100mL 이상: 면허정지 (1년)
• 30 mg/100mL 이상: 운전 위험 수준
• 10 mg/100mL 이하: 완전 회복 상태

⚠️ 중요한 주의사항:
• 이 계산기는 참고용이며 개인차가 있을 수 있습니다
• 실제 음주운전은 절대 금지입니다
• 안전을 위해 음주 후에는 대중교통을 이용하세요
• 음주량과 개인의 체질에 따라 실제 결과는 다를 수 있습니다
• 의학적 문제가 있는 경우 전문의와 상담하세요

🔬 사용된 수학 모델:
"""

        if model_type == "분수계":
            results += f"""• 분수계 미분방정식 기반 모델
• Caputo 분수계 미분과 Mittag-Leffler 함수 사용
• α = {self.alpha}, β = {self.beta}
• 메모리 효과를 고려한 더 현실적인 모델링"""
        else:
            results += f"""• 고전 two-compartment 모델
• 지수함수 기반 단순 모델
• k1 = {self.k1}, k2 = {self.k2}
• 빠른 계산이 가능한 전통적 방법"""

        self.results_text.insert(1.0, results)

    def plot_main_graph(
        self,
        t_hours,
        bac_array,
        legal_time,
        safe_time,
        recovery_time,
        model_type,
        drinking_time,
    ):
        """Plot detailed BAC graph"""
        self.main_ax.clear()

        # Convert to mg/100mL
        bac_mg = bac_array * 100

        # Plot BAC curve
        self.main_ax.plot(
            t_hours, bac_mg, "b-", linewidth=3, label="혈중알코올농도", alpha=0.8
        )

        # Add threshold lines
        self.main_ax.axhline(
            y=80,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="음주운전 단속기준 (80 mg/100mL)",
        )
        self.main_ax.axhline(
            y=50,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="면허정지 기준 (50 mg/100mL)",
        )
        self.main_ax.axhline(
            y=30,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="안전운전 기준 (30 mg/100mL)",
        )
        self.main_ax.axhline(
            y=10,
            color="blue",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="완전회복 기준 (10 mg/100mL)",
        )

        # Add recovery time markers
        if legal_time:
            self.main_ax.axvline(
                x=legal_time,
                color="orange",
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"운전가능: {legal_time:.1f}h",
            )

        if safe_time:
            self.main_ax.axvline(
                x=safe_time,
                color="green",
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"안전운전: {safe_time:.1f}h",
            )

        if recovery_time:
            self.main_ax.axvline(
                x=recovery_time,
                color="blue",
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"완전회복: {recovery_time:.1f}h",
            )

        # Mark peak
        peak_idx = np.argmax(bac_mg)
        peak_time = t_hours[peak_idx]
        peak_value = bac_mg[peak_idx]

        self.main_ax.plot(
            peak_time,
            peak_value,
            "ro",
            markersize=10,
            label=f"최고점: {peak_value:.1f}mg/100mL",
        )

        # Formatting
        self.main_ax.set_xlabel("시간 (hours)", fontsize=12)
        self.main_ax.set_ylabel("혈중알코올농도 (mg/100mL)", fontsize=12)
        self.main_ax.set_title(
            f"혈중알코올농도 예측 - {model_type} 모델 (음주시작: {drinking_time})",
            fontsize=14,
            fontweight="bold",
        )
        self.main_ax.legend(fontsize=10, loc="upper right")
        self.main_ax.grid(True, alpha=0.3)
        self.main_ax.set_xlim(0, 24)
        self.main_ax.set_ylim(0, max(100, np.max(bac_mg) * 1.1))

        # Add background color zones
        self.main_ax.axhspan(
            80, max(100, np.max(bac_mg) * 1.1), alpha=0.1, color="red", label="위험구간"
        )
        self.main_ax.axhspan(50, 80, alpha=0.1, color="orange")
        self.main_ax.axhspan(30, 50, alpha=0.1, color="yellow")
        self.main_ax.axhspan(10, 30, alpha=0.1, color="lightgreen")
        self.main_ax.axhspan(0, 10, alpha=0.1, color="green")

        self.main_fig.tight_layout()
        self.main_canvas.draw()

    def reset_inputs(self):
        """Reset all input fields to default values"""
        self.gender_var.set("남성")
        self.age_var.set(25)
        self.weight_var.set(70.0)
        self.height_var.set(170.0)
        self.drink_var.set("소주")
        self.volume_var.set(360.0)
        self.abv_var.set(17.0)
        self.time_var.set(datetime.now().strftime("%H:%M"))
        self.model_var.set("분수계")

        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.main_ax.clear()
        self.main_canvas.draw()

        # Update preview
        self.update_preview()

    def save_graph(self):
        """Save the current graph"""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            )
            if filename:
                self.main_fig.savefig(filename, dpi=300, bbox_inches="tight")
                messagebox.showinfo(
                    "저장 완료", f"그래프가 저장되었습니다:\n{filename}"
                )
        except Exception as e:
            messagebox.showerror(
                "저장 오류", f"그래프 저장 중 오류가 발생했습니다:\n{str(e)}"
            )

    def save_results(self):
        """Save the current results"""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("저장 완료", f"결과가 저장되었습니다:\n{filename}")
        except Exception as e:
            messagebox.showerror(
                "저장 오류", f"결과 저장 중 오류가 발생했습니다:\n{str(e)}"
            )

    def compare_models(self):
        """Compare fractional and classical models"""
        try:
            # Get current values
            gender = self.gender_var.get()
            age = self.age_var.get()
            weight = self.weight_var.get()
            volume = self.volume_var.get()
            abv = self.abv_var.get()

            # Calculate TBW ratio
            if gender == "남성":
                tbw_ratio = 0.68 - (age - 25) * 0.001
            else:
                tbw_ratio = 0.55 - (age - 25) * 0.001

            # Calculate initial concentration
            A0 = self.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

            # Time array
            t_hours = np.linspace(0, 24, 1000)

            # Calculate both models
            frac_values = []
            class_values = []

            for t in t_hours:
                _, B_frac = self.fractional_bac_model_corrected(
                    t, A0, self.k1, self.k2, self.alpha, self.beta
                )
                _, B_class = self.classical_bac_model(t, A0, self.k1, self.k2)

                frac_values.append(B_frac * 100)
                class_values.append(B_class * 100)

            # Create comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(
                t_hours, frac_values, "b-", linewidth=3, label="분수계 모델", alpha=0.8
            )
            ax.plot(
                t_hours, class_values, "r--", linewidth=3, label="고전 모델", alpha=0.8
            )

            # Add threshold lines
            ax.axhline(y=80, color="red", linestyle=":", alpha=0.5, label="단속기준")
            ax.axhline(y=50, color="orange", linestyle=":", alpha=0.5, label="면허정지")
            ax.axhline(y=30, color="green", linestyle=":", alpha=0.5, label="안전운전")

            ax.set_xlabel("시간 (hours)", fontsize=12)
            ax.set_ylabel("혈중알코올농도 (mg/100mL)", fontsize=12)
            ax.set_title("분수계 vs 고전 모델 비교", fontsize=14, fontweight="bold")
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror(
                "비교 오류", f"모델 비교 중 오류가 발생했습니다:\n{str(e)}"
            )


def main():
    root = tk.Tk()
    app = EnhancedBACCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
