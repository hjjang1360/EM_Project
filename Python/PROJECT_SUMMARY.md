# 🍺 BAC Calculator 프로젝트 완성 보고서

## 📋 프로젝트 개요
**건국대학교 공업수학1** 과목의 최종 프로젝트로, 분수계 미분방정식을 이용한 혈중알코올농도(BAC) 예측 시스템을 구현했습니다.

## ✅ 완성된 애플리케이션

### 1. 🖥️ Enhanced GUI Calculator (`bac_calculator_enhanced.py`) ⭐ **추천**
- **특징**: 최신 개선된 GUI, 탭 기반 인터페이스, 실시간 미리보기
- **기능**: 
  - 직관적인 정보 입력 패널
  - 실시간 BAC 그래프 미리보기
  - 상세한 결과 분석 탭
  - 모델 비교 기능
  - 결과 저장 (그래프, 텍스트)

### 2. 🖼️ Basic GUI Calculator (`bac_calculator_gui.py`)
- **특징**: 기본 GUI 버전, 간단하고 직관적
- **기능**: 
  - 기본적인 입력 폼
  - 실시간 BAC 그래프
  - 술 종류별 기본값 제공

### 3. 💻 Command Line Calculator (`bac_calculator_simple.py`)
- **특징**: 터미널 기반 대화형 인터페이스
- **기능**:
  - 단계별 입력 가이드
  - matplotlib 그래프 출력
  - 반복 계산 가능

### 4. 🌐 Web Calculator (`bac_calculator_web.py`)
- **특징**: Flask 기반 웹 애플리케이션
- **기능**:
  - 반응형 웹 디자인
  - 브라우저에서 바로 사용
  - 실시간 그래프 생성
  - 모바일 친화적

### 5. 🚀 통합 런처 (`launcher.py`)
- **특징**: 모든 애플리케이션을 통합 관리
- **기능**:
  - 의존성 자동 확인 및 설치
  - 애플리케이션 선택 메뉴
  - 시스템 정보 표시

## 🧮 구현된 수학 모델

### 분수계 BAC 모델 (권장)
```
위 농도: A(t) = A₀ × E_α(-k₁t^α)
혈중 농도: B(t) = (A₀k₁)/(k₂-k₁) × [E_α(-k₁t^α) - E_β(-k₂t^β)]
```
- Mittag-Leffler 함수 사용
- 메모리 효과 고려
- α = 0.8, β = 0.9

### 고전 BAC 모델
```
위 농도: A(t) = A₀ × e^(-k₁t)
혈중 농도: B(t) = (A₀k₁)/(k₂-k₁) × [e^(-k₁t) - e^(-k₂t)]
```
- 단순한 지수 감소
- k₁ = 0.8, k₂ = 1.0

## 📊 주요 기능

### 개인정보 입력
- ✅ 성별 (남성/여성)
- ✅ 나이 (19-100세)
- ✅ 몸무게 (30-200kg)
- ✅ 키 (선택사항)

### 음주정보 입력
- ✅ 술 종류 선택 (맥주, 소주, 와인, 위스키, 막걸리, 직접입력)
- ✅ 음주량 (mL)
- ✅ 알코올 도수 (%)
- ✅ 음주 시작 시간

### 계산 결과
- ✅ 초기 농도 (A₀)
- ✅ 최고 BAC 및 도달 시간
- ✅ 회복 시간 예측:
  - 운전 가능 (50mg/100mL)
  - 안전 운전 (30mg/100mL)  
  - 완전 회복 (10mg/100mL)

### 그래프 기능
- ✅ 실시간 BAC 변화 곡선
- ✅ 법적 기준선 표시
- ✅ 회복 시간 마커
- ✅ 최고점 표시
- ✅ 위험 구간 색상 구분

## 🛠️ 기술 스택

### Core Libraries
- **NumPy**: 수치 계산
- **SciPy**: 특수 함수 (gamma, Mittag-Leffler)
- **Matplotlib**: 그래프 생성

### GUI Framework
- **Tkinter**: GUI 애플리케이션
- **ttk**: 스타일 개선

### Web Framework
- **Flask**: 웹 애플리케이션
- **HTML/CSS/JavaScript**: 프론트엔드

## 🎯 사용법

### 빠른 시작
```bash
# 1. 통합 런처 실행 (권장)
python launcher.py

# 2. 원하는 애플리케이션 선택
# 1: Enhanced GUI (추천)
# 2: Basic GUI
# 3: Command Line
# 4: Web App
# 5: Quick Test
```

### 직접 실행
```bash
# Enhanced GUI 실행
python bac_calculator_enhanced.py

# 웹 애플리케이션 실행
python bac_calculator_web.py
# 브라우저에서 http://localhost:5000 접속
```

## ⚠️ 주의사항

### 법적 고지
- 이 계산기는 **교육 및 연구 목적**으로만 사용
- 실제 음주운전은 **절대 금지**
- 개인차가 있을 수 있어 참고용으로만 활용
- 음주 후에는 **대중교통 이용** 권장

### 한국 법적 기준
- **80 mg/100mL 이상**: 음주운전 단속 대상
- **50 mg/100mL 이상**: 면허정지 (1년)
- **30 mg/100mL**: 안전운전 권장 기준
- **10 mg/100mL**: 완전 회복 기준

## 📁 파일 구조

```
BAC_Calculator/
├── launcher.py                    # 통합 실행 도구
├── bac_calculator_enhanced.py     # 개선된 GUI 앱
├── bac_calculator_gui.py          # 기본 GUI 앱  
├── bac_calculator_simple.py       # 명령줄 앱
├── bac_calculator_web.py          # 웹 앱
├── quick_test.py                  # 빠른 테스트
├── demo_test.py                   # 데모 테스트
├── README.md                      # 프로젝트 문서
├── USER_MANUAL.md                 # 사용자 매뉴얼
└── Final/
    └── plot_corrected_comprehensive.py  # 최종 수정된 모델
```

## 🔬 핵심 기술적 성과

### 1. Mittag-Leffler 함수 안정적 구현
```python
def ml1_stable(z, alpha, max_terms=100, tol=1e-15):
    # 수치적으로 안정적인 구현
    # 재귀 관계식 사용으로 오버플로우 방지
```

### 2. 분수계 미분방정식 정확한 해법
```python
def fractional_bac_model_corrected(t, A0, k1, k2, alpha=0.8, beta=0.9):
    # 이론적으로 올바른 분수계 BAC 모델
    # Caputo 분수계 미분방정식 기반
```

### 3. 사용자 친화적 인터페이스
- 직관적인 입력 폼
- 실시간 미리보기
- 상세한 결과 해석
- 다양한 플랫폼 지원

## 🎉 프로젝트 결론

이 프로젝트는 **분수계 미분방정식의 실용적 응용**을 보여주는 성공적인 사례입니다:

1. **수학적 정확성**: 이론적으로 올바른 분수계 모델 구현
2. **실용적 가치**: 실제 사용 가능한 BAC 계산 도구
3. **사용자 경험**: 다양한 인터페이스로 접근성 향상
4. **교육적 효과**: 수학 이론의 현실 적용 체험

### 향후 개선 방향
- 더 많은 개인 변수 고려 (체지방률, 건강상태)
- 음식 섭취 효과 모델링
- 실험 데이터와의 검증
- 모바일 앱 개발

---

**프로젝트 완료일**: 2025년 6월 11일  
**개발자**: 건국대학교 공업수학1 프로젝트팀  
**GitHub**: [프로젝트 링크]  

> "수학은 실세계 문제를 해결하는 강력한 도구입니다. 이 프로젝트가 분수계 미분방정식의 아름다움과 실용성을 보여주기를 바랍니다." 🍺📊
