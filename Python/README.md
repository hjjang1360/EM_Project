# 🍺 BAC Calculator - 혈중알코올농도 계산기 v2.0 (Fixed)

이 프로젝트는 **건국대학교 공업수학1** 과목의 프로젝트로, 분수계 미분방정식을 이용한 혈중알코올농도(BAC) 예측 모델을 구현한 애플리케이션입니다.

## 🔧 v2.0 주요 수정사항

### ✅ 회복 시간 예측 로직 개선
- **문제**: 초기 시점(t=0)에서 BAC=0을 "회복"으로 잘못 인식
- **해결**: 피크 이후 시점만 고려하여 정확한 회복 시간 예측
- **결과**: 현실적이고 신뢰할 수 있는 예측 제공

### ✅ 한글 폰트 표시 개선  
- **문제**: matplotlib 그래프에서 한글이 네모박스로 표시
- **해결**: 한글 폰트 자동 감지 및 설정
- **결과**: 그래프에서 한글 텍스트 정상 표시

## 🎯 빠른 시작 가이드

### 1단계: 프로젝트 다운로드 확인
현재 디렉토리에 다음 파일들이 있는지 확인하세요:
- `launcher_updated.py` - 업데이트된 통합 실행 도구 ⭐
- `bac_calculator_web_fixed.py` - 수정된 웹 앱 ⭐ (신규)
- `bac_calculator_enhanced.py` - 개선된 GUI 앱 (수정됨)
- `bac_calculator_gui.py` - 기본 GUI 앱 (수정됨)
- `bac_calculator_simple.py` - 명령줄 앱 (수정됨)
- `bac_calculator_web.py` - 웹 앱 (수정됨)

### 2단계: 업데이트된 런처 실행 (권장)
```bash
python launcher_updated.py
```

### 3단계: 원하는 애플리케이션 선택
- **수정된 웹 앱**: 가장 최신 기능 포함 ⭐
- **Enhanced GUI**: 고급 기능 포함
- **기본 GUI**: 간단한 인터페이스
- **명령줄 버전**: 빠른 계산
- **테스트 도구**: 수정사항 검증

### 또는 직접 실행
```bash
# 수정된 웹 앱 (가장 추천)
python bac_calculator_web_fixed.py

# 최신 GUI 버전
python bac_calculator_enhanced.py

# 기본 GUI 버전
python bac_calculator_gui.py

# 명령줄 버전
python bac_calculator_simple.py

# 웹 버전 (Flask 설치 필요)
python bac_calculator_web.py
```

## 📋 프로젝트 개요

### 🎯 목적
- 사용자의 개인정보(성별, 나이, 몸무게)와 음주량을 입력받아 혈중알코올농도를 예측
- 언제 술이 깨는지(회복 시간) 계산
- 고전 모델과 분수계 모델의 비교 분석
- 실용적인 BAC 계산 도구 제공

### 🧮 수학적 배경
- **고전 모델**: 지수함수 기반 two-compartment 모델
- **분수계 모델**: Caputo 분수계 미분방정식과 Mittag-Leffler 함수 사용
- 메모리 효과를 고려한 더 현실적인 모델링

## 🚀 애플리케이션 종류

### 1. Enhanced GUI 애플리케이션 (`bac_calculator_enhanced.py`) ⭐ **추천**
**최신 개선된 GUI 버전**

**특징:**
- 현대적이고 직관적인 사용자 인터페이스
- 탭 기반 정보 입력 및 결과 분석
- 실시간 미리보기 그래프
- 상세한 계산 결과 및 텍스트 출력
- 모델 비교 기능
- 결과 저장 기능 (그래프, 텍스트)

**실행 방법:**
```bash
python bac_calculator_enhanced.py
```

### 2. 기본 GUI 애플리케이션 (`bac_calculator_gui.py`)

**특징:**
- 기본적인 그래픽 사용자 인터페이스
- 실시간 BAC 그래프 표시
- 상세한 결과 텍스트 출력
- 술 종류별 기본값 제공

**실행 방법:**
```bash
python bac_calculator_gui.py
```

### 3. 커맨드라인 애플리케이션 (`bac_calculator_simple.py`)

**특징:**
- 대화형 명령줄 인터페이스
- 단계별 입력 가이드
- matplotlib 그래프 출력
- 반복 계산 가능

**실행 방법:**
```bash
python bac_calculator_simple.py
```

### 4. 웹 애플리케이션 (`bac_calculator_web.py`) 🌐 **신규**
**Flask 기반 웹 인터페이스**

**특징:**
- 반응형 웹 디자인
- 실시간 그래프 생성
- 브라우저에서 바로 사용 가능
- 모바일 친화적 인터페이스

**실행 방법:**
```bash
# Flask 설치 필요: pip install flask
python bac_calculator_web.py
# 브라우저에서 http://localhost:5000 접속
```

### 5. Streamlit 웹 애플리케이션 (`bac_calculator_app.py`)

**특징:**
- Streamlit 기반 웹 인터페이스
- 반응형 그래프 (Plotly)
- 실시간 결과 업데이트

**실행 방법:**
```bash
streamlit run bac_calculator_app.py
```

## 📊 기능

### 🔧 주요 기능
1. **개인정보 입력**
   - 성별 (남성/여성)
   - 나이 (체수분 비율 계산에 사용)
   - 몸무게
   - 키 (선택적)

2. **음주정보 입력**
   - 술 종류 (맥주, 소주, 와인, 위스키, 막걸리, 직접입력)
   - 음주량 (mL)
   - 알코올 도수 (%)
   - 음주 시작 시간

3. **모델 선택**
   - 분수계 모델 (메모리 효과 포함, 더 정확)
   - 고전 모델 (단순한 지수 감소)

4. **결과 출력**
   - 최고 BAC 및 도달 시간
   - 운전 가능 시간 (50mg/100mL 이하)
   - 안전 운전 시간 (30mg/100mL 이하)
   - 완전 회복 시간 (10mg/100mL 이하)
   - BAC 변화 그래프

### 📈 그래프 기능
- 시간별 BAC 변화 곡선
- 법적 기준선 표시 (한국 기준)
- 회복 시간 마커
- 인터랙티브 그래프 (웹 버전)

## 🔬 이론적 배경

### 고전 Two-Compartment 모델
```
dA/dt = -k₁ * A(t)
dB/dt = k₁ * A(t) - k₂ * B(t)
```

**해:**
- A(t) = A₀ * exp(-k₁t)
- B(t) = (A₀k₁)/(k₂-k₁) * [exp(-k₁t) - exp(-k₂t)]

### 분수계 모델 (Caputo 미분)
```
D^α A(t) = -k₁ * A(t)
D^β B(t) = k₁ * A(t) - k₂ * B(t)
```

**해:**
- A(t) = A₀ * E_α(-k₁t^α)
- B(t) = (A₀k₁)/(k₂-k₁) * [E_α(-k₁t^α) - E_β(-k₂t^β)]

여기서 E_α는 Mittag-Leffler 함수입니다.

### 매개변수
- **k₁ = 0.8 h⁻¹**: 위에서 혈액으로의 흡수율
- **k₂ = 1.0 h⁻¹**: 혈액에서의 제거율
- **α = 0.8**: 흡수 과정의 분수 차수
- **β = 0.9**: 제거 과정의 분수 차수

## 💻 설치 및 실행

### 필요 패키지
```bash
pip install numpy matplotlib scipy tkinter
# 웹 버전 사용시 추가:
pip install streamlit plotly
```

### 실행 방법

1. **GUI 버전 (권장)**
   ```bash
   python bac_calculator_gui.py
   ```

2. **커맨드라인 버전**
   ```bash
   python bac_calculator_simple.py
   ```

3. **웹 버전**
   ```bash
   streamlit run bac_calculator_app.py
   ```

## 🎮 사용 방법

### GUI 애플리케이션 사용법

1. **개인정보 입력**
   - 성별, 나이, 몸무게, 키 입력

2. **음주정보 입력**
   - 술 종류 선택 (자동으로 도수와 기본량 설정)
   - 음주량 조정
   - 음주 시작 시간 입력

3. **계산 실행**
   - "🧮 BAC 계산하기" 버튼 클릭

4. **결과 확인**
   - 왼쪽: 상세 텍스트 결과
   - 오른쪽: BAC 변화 그래프

### 커맨드라인 사용법

1. 프로그램 실행
2. 단계별 입력 프롬프트 따라하기
3. 그래프 표시 여부 선택
4. 재계산 여부 선택

## ⚠️ 주의사항

### 🚨 중요 경고
- **이 계산기는 참고용이며 개인차가 있을 수 있습니다**
- **실제 음주운전은 절대 금지입니다**
- **안전을 위해 음주 후에는 대중교통을 이용하세요**

### 📏 한국 음주운전 기준
- **면허정지**: 50-80 mg/100mL
- **면허취소**: 80 mg/100mL 이상
- **안전 기준**: 30 mg/100mL 이하 권장

### 🔬 모델의 한계
- 개인의 대사율 차이 고려 안됨
- 공복/식후 상태 미반영
- 음주 속도 고려 안됨
- 의학적 상태 미반영

## 📁 파일 구조

```
BAC_Calculator/
├── bac_calculator_gui.py          # GUI 애플리케이션
├── bac_calculator_simple.py       # 커맨드라인 애플리케이션
├── bac_calculator_app.py          # 웹 애플리케이션
├── plot_corrected_comprehensive.py # 이론적 분석 스크립트
├── fractional_bac_corrected.py    # 수정된 분수계 모델
└── README.md                      # 이 파일
```

## 🔬 연구 결과

### 모델 비교
- **분수계 모델**: 메모리 효과로 인한 완만한 변화
- **고전 모델**: 빠른 흡수와 제거, 높은 피크

### 실용적 차이
- 회복 시간에서 유의미한 차이
- 분수계 모델이 더 보수적(안전한) 예측
- 개인차를 더 잘 반영

## 👥 개발팀

- **프로젝트**: 건국대학교 공업수학1
- **주제**: 분수계 미분방정식을 이용한 BAC 모델링
- **개발**: 프로젝트 G2팀

## 📜 라이선스

이 프로젝트는 교육용으로 개발되었습니다.

---

*Made with ❤️ for Engineering Mathematics 1 Project*
