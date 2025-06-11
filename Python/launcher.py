#!/usr/bin/env python3
"""
BAC Calculator Launcher - 통합 실행 스크립트
사용할 애플리케이션을 선택하여 실행할 수 있는 런처
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
        },
        "2": {
            "name": "Basic GUI Calculator",
            "file": "bac_calculator_gui.py",
            "description": "기본 GUI 버전",
        },
        "3": {
            "name": "Command Line Calculator",
            "file": "bac_calculator_simple.py",
            "description": "명령줄 버전",
        },
        "4": {
            "name": "Web Calculator",
            "file": "bac_calculator_web.py",
            "description": "Flask 웹 버전",
        },
        "5": {
            "name": "Quick Test",
            "file": "quick_test.py",
            "description": "빠른 테스트",
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

    try:
        if app_choice == "4":  # Web app
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
    print("\n" + "=" * 60)
    print("🍺 BAC Calculator - 애플리케이션 선택")
    print("=" * 60)
    print()
    print("1. 🖥️  Enhanced GUI Calculator (권장)")
    print("   └─ 최신 개선된 GUI, 탭 기반 인터페이스, 실시간 미리보기")
    print()
    print("2. 🖼️  Basic GUI Calculator")
    print("   └─ 기본 GUI 버전, 간단한 인터페이스")
    print()
    print("3. 💻 Command Line Calculator")
    print("   └─ 터미널 기반, 대화형 입력")
    print()
    print("4. 🌐 Web Calculator")
    print("   └─ 브라우저 기반, Flask 웹 애플리케이션")
    print()
    print("5. 🧪 Quick Test")
    print("   └─ 빠른 기능 테스트")
    print()
    print("0. ❌ 종료")
    print()
    print("=" * 60)


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
        "quick_test.py",
    ]

    print("\n📁 파일 상태:")
    for file in core_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")


def main():
    """메인 함수"""
    print("🍺 BAC Calculator Launcher")
    print("건국대학교 공업수학1 프로젝트")
    print("혈중알코올농도 계산기 통합 실행 도구")

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

        choice = input("선택하세요 (0-5): ").strip()

        if choice == "0":
            print("\n프로그램을 종료합니다. 감사합니다! 🙏")
            break
        elif choice in ["1", "2", "3", "4", "5"]:
            success = run_application(choice)
            if success:
                print("\n애플리케이션이 정상적으로 종료되었습니다.")
            else:
                print("\n애플리케이션 실행에 문제가 있었습니다.")

            input("\nEnter를 눌러서 메뉴로 돌아가세요...")
        else:
            print("올바른 번호를 선택해주세요 (0-5)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        print("문제가 지속되면 개발자에게 문의하세요.")
