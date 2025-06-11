#!/usr/bin/env python3
"""
Updated BAC Calculator Launcher - 수정된 통합 실행 스크립트
회복 시간 및 폰트 문제 수정사항을 포함한 런처
"""

import subprocess
import sys
import os
from datetime import datetime


def check_dependencies():
    """필요한 라이브러리 설치 확인"""
    required_packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "tkinter": None,  # Built-in
        "flask": "flask",
    }

    missing_packages = []

    for package, install_name in required_packages.items():
        try:
            if package == "tkinter":
                import tkinter
            else:
                __import__(package)
        except ImportError:
            if install_name:
                missing_packages.append(install_name)

    return missing_packages


def install_missing_packages(packages):
    """누락된 패키지 설치"""
    if not packages:
        return True

    print(f"누락된 패키지를 설치합니다: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ 패키지 설치 완료!")
        return True
    except subprocess.CalledProcessError:
        print("❌ 패키지 설치 실패!")
        return False


def run_application(app_choice):
    """선택된 애플리케이션 실행"""
    apps = {
        "1": {
            "name": "Enhanced GUI Calculator",
            "file": "bac_calculator_enhanced.py",
            "description": "최신 개선된 GUI 버전 (권장)",
            "note": "",
        },
        "2": {
            "name": "Basic GUI Calculator",
            "file": "bac_calculator_gui.py",
            "description": "기본 GUI 버전",
            "note": "",
        },
        "3": {
            "name": "Command Line Calculator",
            "file": "bac_calculator_simple.py",
            "description": "명령줄 버전",
            "note": "",
        },
        "4": {
            "name": "Web Calculator (Original)",
            "file": "bac_calculator_web.py",
            "description": "Flask 웹 버전 (원본)",
            "note": "⚠️ 회복시간 및 폰트 문제 있음",
        },
        "5": {
            "name": "Web Calculator (Fixed) ⭐",
            "file": "bac_calculator_web_fixed.py",
            "description": "Flask 웹 버전 (수정됨)",
            "note": "✅ 회복시간 및 폰트 문제 해결",
        },
        "6": {
            "name": "Recovery Time Test",
            "file": "simple_recovery_test.py",
            "description": "회복시간 로직 테스트",
            "note": "🧪 수정사항 검증용",
        },
    }

    if app_choice not in apps:
        print("올바른 선택이 아닙니다.")
        return False

    app = apps[app_choice]

    if not os.path.exists(app["file"]):
        print(f"❌ 파일을 찾을 수 없습니다: {app['file']}")
        return False

    print(f"\n🚀 {app['name']} 실행 중...")
    print(f"📝 {app['description']}")
    if app["note"]:
        print(f"💡 {app['note']}")

    try:
        if app_choice in ["4", "5"]:  # Web apps
            print("\n🌐 웹 서버를 시작합니다...")
            print("브라우저에서 http://localhost:5000 으로 접속하세요")
            print("종료하려면 Ctrl+C를 누르세요")

        subprocess.run([sys.executable, app["file"]])
        return True

    except KeyboardInterrupt:
        print("\n\n애플리케이션이 사용자에 의해 종료되었습니다.")
        return True
    except FileNotFoundError:
        print(f"❌ Python 인터프리터를 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        return False


def show_menu():
    """메뉴 표시"""
    print("\n" + "=" * 70)
    print("🍺 BAC Calculator - 애플리케이션 선택 (업데이트됨)")
    print("=" * 70)
    print()
    print("📱 GUI 애플리케이션")
    print("1. 🖥️  Enhanced GUI Calculator (권장)")
    print("   └─ 최신 개선된 GUI, 탭 기반 인터페이스, 실시간 미리보기")
    print()
    print("2. 🖼️  Basic GUI Calculator")
    print("   └─ 기본 GUI 버전, 간단한 인터페이스")
    print()
    print("💻 명령줄 애플리케이션")
    print("3. 🔤 Command Line Calculator")
    print("   └─ 터미널 기반, 대화형 입력")
    print()
    print("🌐 웹 애플리케이션")
    print("4. 🌍 Web Calculator (Original)")
    print("   └─ 브라우저 기반, Flask 웹 애플리케이션 ⚠️ 문제 있음")
    print()
    print("5. 🌟 Web Calculator (Fixed) ⭐ 추천")
    print("   └─ 수정된 웹 버전 (회복시간 및 폰트 문제 해결)")
    print()
    print("🧪 테스트 도구")
    print("6. 🔬 Recovery Time Test")
    print("   └─ 회복시간 로직 수정사항 검증")
    print()
    print("0. ❌ 종료")
    print()
    print("=" * 70)


def show_system_info():
    """시스템 정보 표시"""
    print("📋 시스템 정보")
    print("-" * 30)
    print(f"Python 버전: {sys.version}")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"현재 디렉토리: {os.getcwd()}")

    # Check if core files exist
    core_files = [
        "bac_calculator_enhanced.py",
        "bac_calculator_gui.py",
        "bac_calculator_simple.py",
        "bac_calculator_web.py",
        "bac_calculator_web_fixed.py",
        "simple_recovery_test.py",
    ]

    print("\n📁 파일 상태:")
    for file in core_files:
        status = "✅" if os.path.exists(file) else "❌"
        if file == "bac_calculator_web_fixed.py" and status == "✅":
            print(f"  {status} {file} ⭐ (수정 버전)")
        else:
            print(f"  {status} {file}")


def show_fixes():
    """수정사항 표시"""
    print("\n🔧 주요 수정사항:")
    print("-" * 40)
    print("✅ 회복 시간 예측 로직 개선")
    print("   - 피크 이후 시점만 고려하여 정확한 예측")
    print("   - 초기 시점(t=0) 오인식 문제 해결")
    print()
    print("✅ 한글 폰트 표시 문제 해결")
    print("   - Windows 환경에서 한글 폰트 자동 감지")
    print("   - 그래프 라벨 네모 표시 문제 수정")
    print()
    print("✅ 사용자 경험 개선")
    print("   - 더 직관적인 오류 메시지")
    print("   - 개선된 시각적 피드백")


def main():
    """메인 함수"""
    print("🍺 BAC Calculator Launcher (Updated)")
    print("건국대학교 공업수학1 프로젝트")
    print("혈중알코올농도 계산기 통합 실행 도구 - 수정 버전")

    # Show fixes
    show_fixes()

    # Show system info
    show_system_info()

    # Check dependencies
    print("\n🔍 의존성 확인 중...")
    missing = check_dependencies()

    if missing:
        print(f"❌ 누락된 패키지: {', '.join(missing)}")
        install_choice = (
            input("패키지를 자동으로 설치하시겠습니까? (y/n): ").strip().lower()
        )

        if install_choice in ["y", "yes", "예", "ㅇ"]:
            if not install_missing_packages(missing):
                print("패키지 설치에 실패했습니다. 수동으로 설치해주세요:")
                for pkg in missing:
                    print(f"  pip install {pkg}")
                return
        else:
            print("필요한 패키지를 먼저 설치해주세요.")
            return
    else:
        print("✅ 모든 의존성이 충족되었습니다!")

    # Main menu loop
    while True:
        show_menu()

        choice = input("선택하세요 (0-6): ").strip()

        if choice == "0":
            print("\n프로그램을 종료합니다. 감사합니다! 🙏")
            break
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            success = run_application(choice)
            if success:
                print("\n애플리케이션이 정상적으로 종료되었습니다.")
            else:
                print("\n애플리케이션 실행에 문제가 있었습니다.")

            input("\nEnter를 눌러서 메뉴로 돌아가세요...")
        else:
            print("올바른 번호를 선택해주세요 (0-6)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        print("문제가 지속되면 개발자에게 문의하세요.")
