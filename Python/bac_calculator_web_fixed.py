#!/usr/bin/env python3
"""
Fixed Web BAC Calculator using Flask
웹 브라우저에서 사용할 수 있는 간단한 BAC 계산기 (회복시간 및 한글폰트 수정)
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.special import gamma
import json
from datetime import datetime, timedelta
import base64
import io
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Configure matplotlib for Korean fonts
try:
    # Windows에서 한글 폰트 설정
    font_list = [font.name for font in fm.fontManager.ttflist]
    korean_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim", "Dotum"]
    available_font = None

    for font in korean_fonts:
        if font in font_list:
            available_font = font
            break

    if available_font:
        plt.rcParams["font.family"] = available_font
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

except:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

app = Flask(__name__)


class WebBACCalculator:
    def __init__(self):
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


# Global calculator instance
calculator = WebBACCalculator()


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate_bac():
    """Calculate BAC and return results"""
    try:
        data = request.json

        # Extract input data
        gender = data["gender"]
        age = int(data["age"])
        weight = float(data["weight"])
        volume = float(data["volume"])
        abv = float(data["abv"])
        model_type = data["model_type"]
        drinking_time = data["drinking_time"]

        # Calculate TBW ratio
        if gender == "male":
            tbw_ratio = 0.68 - (age - 25) * 0.001
        else:
            tbw_ratio = 0.55 - (age - 25) * 0.001

        # Calculate initial concentration
        A0 = calculator.calculate_initial_concentration(weight, tbw_ratio, volume, abv)

        # Time array (0 to 24 hours)
        t_hours = np.linspace(0, 24, 1000)
        bac_values = []

        for t in t_hours:
            if model_type == "fractional":
                _, B_t = calculator.fractional_bac_model_corrected(
                    t,
                    A0,
                    calculator.k1,
                    calculator.k2,
                    calculator.alpha,
                    calculator.beta,
                )
            else:
                _, B_t = calculator.classical_bac_model(
                    t, A0, calculator.k1, calculator.k2
                )

            bac_values.append(B_t)

        bac_array = np.array(bac_values)

        # Find recovery times with improved logic
        legal_time, safe_time, recovery_time = calculator.find_recovery_times(
            t_hours, bac_array
        )

        # Calculate statistics
        peak_bac = np.max(bac_array) * 100
        peak_time = t_hours[np.argmax(bac_array)]

        # Generate graph with Korean font support
        graph_url = generate_graph(
            t_hours, bac_array, legal_time, safe_time, recovery_time, model_type
        )

        # Prepare results
        results = {
            "success": True,
            "input_data": {
                "gender": "남성" if gender == "male" else "여성",
                "age": age,
                "weight": weight,
                "volume": volume,
                "abv": abv,
                "model_type": "분수계" if model_type == "fractional" else "고전",
                "drinking_time": drinking_time,
                "tbw_ratio": round(tbw_ratio, 3),
                "alcohol_mass": round(volume * abv / 100 * 0.789, 1),
            },
            "calculations": {
                "A0": round(A0, 3),
                "peak_bac": round(peak_bac, 1),
                "peak_time": round(peak_time, 1),
            },
            "recovery_times": {
                "legal_time": round(legal_time, 1) if legal_time else None,
                "safe_time": round(safe_time, 1) if safe_time else None,
                "recovery_time": round(recovery_time, 1) if recovery_time else None,
            },
            "graph_url": graph_url,
            "chart_data": {"time": t_hours.tolist(), "bac": (bac_array * 100).tolist()},
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def generate_graph(
    t_hours, bac_array, legal_time, safe_time, recovery_time, model_type
):
    """Generate BAC graph and return as base64 string - WITH KOREAN FONT SUPPORT"""
    try:
        plt.figure(figsize=(12, 8))

        # Convert to mg/100mL
        bac_mg = bac_array * 100

        # Plot BAC curve
        plt.plot(t_hours, bac_mg, "b-", linewidth=3, label="혈중알코올농도", alpha=0.8)

        # Add threshold lines with Korean labels
        plt.axhline(
            y=80,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="음주운전 단속기준 (80 mg/100mL)",
        )
        plt.axhline(
            y=50,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="면허정지 기준 (50 mg/100mL)",
        )
        plt.axhline(
            y=30,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="안전운전 기준 (30 mg/100mL)",
        )
        plt.axhline(
            y=10,
            color="blue",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="완전회복 기준 (10 mg/100mL)",
        )

        # Add recovery time markers
        if legal_time:
            plt.axvline(
                x=legal_time,
                color="orange",
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"운전가능: {legal_time:.1f}h",
            )

        if safe_time:
            plt.axvline(
                x=safe_time,
                color="green",
                linestyle=":",
                alpha=0.8,
                linewidth=3,
                label=f"안전운전: {safe_time:.1f}h",
            )

        if recovery_time:
            plt.axvline(
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

        plt.plot(
            peak_time,
            peak_value,
            "ro",
            markersize=10,
            label=f"최고점: {peak_value:.1f}mg/100mL",
        )

        # Formatting with Korean support
        plt.xlabel("시간 (hours)", fontsize=12)
        plt.ylabel("혈중알코올농도 (mg/100mL)", fontsize=12)
        plt.title(
            f'혈중알코올농도 예측 - {"분수계" if model_type == "fractional" else "고전"} 모델',
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        plt.ylim(0, max(100, np.max(bac_mg) * 1.1))

        plt.tight_layout()

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{graph_url}"

    except Exception as e:
        print(f"Graph generation error: {e}")
        return None


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    import os

    os.makedirs("templates", exist_ok=True)

    # Create the HTML template (same as before)
    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍺 BAC Calculator - 웹 버전 (수정)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            display: flex;
            min-height: 600px;
        }
        
        .input-panel {
            flex: 1;
            padding: 30px;
            background: #f8f9fa;
        }
        
        .results-panel {
            flex: 2;
            padding: 30px;
            background: white;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ecf0f1;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .results h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .result-item:last-child {
            border-bottom: none;
        }
        
        .result-value {
            font-weight: bold;
            color: #e74c3c;
        }
        
        .graph-container {
            text-align: center;
            margin-top: 20px;
        }
        
        .graph-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
            color: #856404;
        }
        
        .warning h4 {
            margin-bottom: 10px;
            color: #e17055;
        }
        
        .fix-notice {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            color: #155724;
        }
        
        @media (max-width: 768px) {
            .content {
                flex-direction: column;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍺 혈중알코올농도(BAC) 계산기</h1>
            <p>분수계 미분방정식을 이용한 정확한 BAC 예측 시스템 (개선 버전)</p>
        </div>
        
        <div class="content">
            <div class="input-panel">
                <div class="fix-notice">
                    <h4>🔧 수정 사항</h4>
                    <ul>
                        <li>회복 시간 예측 로직 개선</li>
                        <li>한글 폰트 표시 문제 해결</li>
                        <li>피크 이후 시점만 고려하여 정확한 예측</li>
                    </ul>
                </div>
                
                <h2>📝 정보 입력</h2>
                
                <div class="form-group">
                    <label for="gender">성별</label>
                    <select id="gender">
                        <option value="male">남성</option>
                        <option value="female">여성</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="age">나이</label>
                    <input type="number" id="age" min="19" max="100" value="25">
                </div>
                
                <div class="form-group">
                    <label for="weight">몸무게 (kg)</label>
                    <input type="number" id="weight" min="30" max="200" step="0.5" value="70">
                </div>
                
                <div class="form-group">
                    <label for="drink_type">술 종류</label>
                    <select id="drink_type" onchange="updateDrinkPreset()">
                        <option value="beer">맥주 (5%, 500mL)</option>
                        <option value="soju" selected>소주 (17%, 360mL)</option>
                        <option value="wine">와인 (12%, 150mL)</option>
                        <option value="whiskey">위스키 (40%, 50mL)</option>
                        <option value="makgeolli">막걸리 (6%, 300mL)</option>
                        <option value="custom">직접입력</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="volume">음주량 (mL)</label>
                    <input type="number" id="volume" min="10" max="2000" value="360">
                </div>
                
                <div class="form-group">
                    <label for="abv">알코올 도수 (%)</label>
                    <input type="number" id="abv" min="0.5" max="60" step="0.1" value="17">
                </div>
                
                <div class="form-group">
                    <label for="drinking_time">음주 시작 시간</label>
                    <input type="time" id="drinking_time">
                </div>
                
                <div class="form-group">
                    <label for="model_type">계산 모델</label>
                    <select id="model_type">
                        <option value="fractional" selected>분수계 모델 (정확, 권장)</option>
                        <option value="classical">고전 모델 (단순, 빠름)</option>
                    </select>
                </div>
                
                <button class="btn" onclick="calculateBAC()">🧮 BAC 계산하기</button>
            </div>
            
            <div class="results-panel">
                <div id="loading" style="display: none; text-align: center; padding: 50px;">
                    <h3>계산 중...</h3>
                    <p>잠시만 기다려주세요.</p>
                </div>
                
                <div id="results" style="display: none;">
                    <div class="results">
                        <h3>📊 계산 결과</h3>
                        <div id="calculation-results"></div>
                    </div>
                    
                    <div class="results">
                        <h3>⏰ 회복 시간 예측 (개선됨)</h3>
                        <div id="recovery-results"></div>
                    </div>
                    
                    <div class="graph-container">
                        <img id="graph" alt="BAC Graph" style="display: none;">
                    </div>
                </div>
                
                <div class="warning">
                    <h4>⚠️ 중요한 주의사항</h4>
                    <ul>
                        <li>이 계산기는 참고용이며 개인차가 있을 수 있습니다</li>
                        <li>실제 음주운전은 절대 금지입니다</li>
                        <li>안전을 위해 음주 후에는 대중교통을 이용하세요</li>
                        <li>의학적 문제가 있는 경우 전문의와 상담하세요</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Set current time as default
        document.getElementById('drinking_time').value = new Date().toTimeString().slice(0,5);
        
        // Drink presets
        const drinkPresets = {
            beer: { abv: 5, volume: 500 },
            soju: { abv: 17, volume: 360 },
            wine: { abv: 12, volume: 150 },
            whiskey: { abv: 40, volume: 50 },
            makgeolli: { abv: 6, volume: 300 },
            custom: { abv: 20, volume: 100 }
        };
        
        function updateDrinkPreset() {
            const drinkType = document.getElementById('drink_type').value;
            const preset = drinkPresets[drinkType];
            
            if (preset) {
                document.getElementById('abv').value = preset.abv;
                document.getElementById('volume').value = preset.volume;
            }
        }
        
        async function calculateBAC() {
            const loadingDiv = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');
            const button = document.querySelector('.btn');
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            button.disabled = true;
            
            // Collect input data
            const data = {
                gender: document.getElementById('gender').value,
                age: document.getElementById('age').value,
                weight: document.getElementById('weight').value,
                volume: document.getElementById('volume').value,
                abv: document.getElementById('abv').value,
                model_type: document.getElementById('model_type').value,
                drinking_time: document.getElementById('drinking_time').value
            };
            
            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result);
                } else {
                    alert('계산 오류: ' + result.error);
                }
            } catch (error) {
                alert('서버 오류: ' + error.message);
            } finally {
                // Hide loading
                loadingDiv.style.display = 'none';
                button.disabled = false;
            }
        }
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            const calcResults = document.getElementById('calculation-results');
            const recoveryResults = document.getElementById('recovery-results');
            const graphImg = document.getElementById('graph');
            
            // Display calculation results
            calcResults.innerHTML = `
                <div class="result-item">
                    <span>초기 농도 (A0):</span>
                    <span class="result-value">${result.calculations.A0} g/L</span>
                </div>
                <div class="result-item">
                    <span>최고 BAC:</span>
                    <span class="result-value">${result.calculations.peak_bac} mg/100mL</span>
                </div>
                <div class="result-item">
                    <span>최고점 도달:</span>
                    <span class="result-value">음주 후 ${result.calculations.peak_time}시간</span>
                </div>
                <div class="result-item">
                    <span>순수 알코올:</span>
                    <span class="result-value">${result.input_data.alcohol_mass}g</span>
                </div>
            `;
            
            // Display recovery results with improved logic indication
            let recoveryHTML = '';
            
            if (result.recovery_times.legal_time) {
                recoveryHTML += `
                    <div class="result-item">
                        <span>🚗 운전 가능 (50mg/100mL):</span>
                        <span class="result-value">음주 후 ${result.recovery_times.legal_time}시간</span>
                    </div>
                `;
            } else {
                recoveryHTML += `
                    <div class="result-item">
                        <span>🚗 운전 가능:</span>
                        <span class="result-value">24시간 내 불가능</span>
                    </div>
                `;
            }
            
            if (result.recovery_times.safe_time) {
                recoveryHTML += `
                    <div class="result-item">
                        <span>✅ 안전 운전 (30mg/100mL):</span>
                        <span class="result-value">음주 후 ${result.recovery_times.safe_time}시간</span>
                    </div>
                `;
            } else {
                recoveryHTML += `
                    <div class="result-item">
                        <span>✅ 안전 운전:</span>
                        <span class="result-value">24시간 내 불가능</span>
                    </div>
                `;
            }
            
            if (result.recovery_times.recovery_time) {
                recoveryHTML += `
                    <div class="result-item">
                        <span>🎉 완전 회복 (10mg/100mL):</span>
                        <span class="result-value">음주 후 ${result.recovery_times.recovery_time}시간</span>
                    </div>
                `;
            } else {
                recoveryHTML += `
                    <div class="result-item">
                        <span>🎉 완전 회복:</span>
                        <span class="result-value">24시간 내 불가능</span>
                    </div>
                `;
            }
            
            recoveryHTML += `
                <div class="result-item" style="margin-top: 10px; font-style: italic; color: #27ae60;">
                    <span>📈 개선된 예측:</span>
                    <span>피크 이후 시점만 고려</span>
                </div>
            `;
            
            recoveryResults.innerHTML = recoveryHTML;
            
            // Display graph
            if (result.graph_url) {
                graphImg.src = result.graph_url;
                graphImg.style.display = 'block';
            }
            
            // Show results
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>"""

    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(html_template)

    print("🌐 개선된 웹 BAC 계산기를 시작합니다...")
    print("🔧 수정 사항:")
    print("  - 회복 시간 예측 로직 개선 (피크 이후 시점만 고려)")
    print("  - 한글 폰트 표시 문제 해결")
    print("브라우저에서 http://localhost:5000 으로 접속하세요")

    app.run(debug=True, host="0.0.0.0", port=5000)
