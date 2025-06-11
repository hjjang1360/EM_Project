#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAC Calculator v2.0 - Updated Launcher
Comprehensive launcher for all BAC calculator applications
All major issues have been fixed in this version.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import webbrowser
import threading
import time


class BACCalculatorLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("BAC Calculator v2.0 - Launcher")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Configure style
        self.setup_style()

        # Create GUI
        self.create_widgets()

        # Web server process
        self.web_process = None

    def setup_style(self):
        """Configure the appearance"""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="BAC Calculator v2.0", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        # Subtitle with fixes info
        subtitle_text = "🎉 All major issues fixed! Korean fonts, recovery logic, and syntax errors resolved."
        subtitle_label = ttk.Label(
            main_frame, text=subtitle_text, style="Subtitle.TLabel"
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 20))

        # Applications section
        apps_frame = ttk.LabelFrame(main_frame, text="📱 Applications", padding="15")
        apps_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        apps_frame.columnconfigure(0, weight=1)

        # Application buttons
        self.create_app_buttons(apps_frame)

        # Tools section
        tools_frame = ttk.LabelFrame(
            main_frame, text="🔧 Testing & Validation Tools", padding="15"
        )
        tools_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        tools_frame.columnconfigure(0, weight=1)

        # Tool buttons
        self.create_tool_buttons(tools_frame)

        # Documentation section
        docs_frame = ttk.LabelFrame(main_frame, text="📚 Documentation", padding="15")
        docs_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        docs_frame.columnconfigure(0, weight=1)

        # Documentation buttons
        self.create_doc_buttons(docs_frame)

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)

        # Status label
        self.status_label = ttk.Label(
            status_frame, text="Ready to launch applications", style="Subtitle.TLabel"
        )
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Version info
        version_label = ttk.Label(
            status_frame, text="v2.0 - Production Ready ✅", style="Subtitle.TLabel"
        )
        version_label.grid(row=0, column=1, sticky=tk.E)

    def create_app_buttons(self, parent):
        """Create application launch buttons"""
        apps = [
            (
                "🖥️ Simple GUI",
                "bac_calculator_simple.py",
                "Basic calculator with clean interface",
            ),
            (
                "🎨 Enhanced GUI",
                "bac_calculator_enhanced.py",
                "Advanced GUI with extra features",
            ),
            ("💻 Complete App", "bac_calculator_app.py", "Full-featured application"),
            (
                "🌐 Web Interface",
                "bac_calculator_web_fixed.py",
                "Browser-based calculator",
            ),
        ]

        for i, (name, file, desc) in enumerate(apps):
            frame = ttk.Frame(parent)
            frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            frame.columnconfigure(1, weight=1)

            btn = ttk.Button(
                frame, text=name, width=20, command=lambda f=file: self.launch_app(f)
            )
            btn.grid(row=0, column=0, padx=(0, 10))

            desc_label = ttk.Label(frame, text=desc)
            desc_label.grid(row=0, column=1, sticky=tk.W)

    def create_tool_buttons(self, parent):
        """Create testing tool buttons"""
        tools = [
            (
                "🧪 Validation Test",
                "validation_test.py",
                "Comprehensive functionality test",
            ),
            (
                "⚡ Quick Test",
                "quick_functionality_test.py",
                "Fast recovery logic test",
            ),
            (
                "📊 Recovery Comparison",
                "test_recovery_fix.py",
                "Compare old vs new logic",
            ),
        ]

        for i, (name, file, desc) in enumerate(tools):
            frame = ttk.Frame(parent)
            frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            frame.columnconfigure(1, weight=1)

            btn = ttk.Button(
                frame, text=name, width=20, command=lambda f=file: self.run_tool(f)
            )
            btn.grid(row=0, column=0, padx=(0, 10))

            desc_label = ttk.Label(frame, text=desc)
            desc_label.grid(row=0, column=1, sticky=tk.W)

    def create_doc_buttons(self, parent):
        """Create documentation buttons"""
        docs = [
            ("📖 User Manual", "USER_MANUAL.md", "How to use the calculator"),
            ("🔧 Fixes Report", "FIXES_COMPLETED.md", "Detailed fix information"),
            (
                "📋 Testing Report",
                "FINAL_TESTING_REPORT.md",
                "Complete testing results",
            ),
            ("📝 README", "README.md", "Project overview and setup"),
        ]

        for i, (name, file, desc) in enumerate(docs):
            frame = ttk.Frame(parent)
            frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            frame.columnconfigure(1, weight=1)

            btn = ttk.Button(
                frame, text=name, width=20, command=lambda f=file: self.open_doc(f)
            )
            btn.grid(row=0, column=0, padx=(0, 10))

            desc_label = ttk.Label(frame, text=desc)
            desc_label.grid(row=0, column=1, sticky=tk.W)

    def launch_app(self, filename):
        """Launch a BAC calculator application"""
        try:
            if not os.path.exists(filename):
                messagebox.showerror("Error", f"File not found: {filename}")
                return

            self.update_status(f"Launching {filename}...")

            if "web" in filename:
                # Launch web server in background
                if self.web_process is None:
                    self.web_process = subprocess.Popen([sys.executable, filename])
                    time.sleep(2)  # Give server time to start
                    webbrowser.open("http://localhost:8080")
                    self.update_status("Web server running at http://localhost:8080")
                else:
                    webbrowser.open("http://localhost:8080")
                    self.update_status("Opening web interface...")
            else:
                # Launch GUI application
                subprocess.Popen([sys.executable, filename])
                self.update_status(f"Launched {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {filename}:\n{str(e)}")
            self.update_status("Ready")

    def run_tool(self, filename):
        """Run a testing tool"""
        try:
            if not os.path.exists(filename):
                messagebox.showerror("Error", f"File not found: {filename}")
                return

            self.update_status(f"Running {filename}...")

            # Run tool in new command window
            if os.name == "nt":  # Windows
                subprocess.Popen(
                    ["cmd", "/c", f"python {filename} & pause"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:  # Unix-like
                subprocess.Popen(["python", filename])

            self.update_status(f"Tool {filename} started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {filename}:\n{str(e)}")
            self.update_status("Ready")

    def open_doc(self, filename):
        """Open documentation file"""
        try:
            if not os.path.exists(filename):
                messagebox.showerror("Error", f"File not found: {filename}")
                return

            # Open with default application
            if os.name == "nt":  # Windows
                os.startfile(filename)
            else:  # Unix-like
                subprocess.call(["open", filename])

            self.update_status(f"Opened {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {filename}:\n{str(e)}")
            self.update_status("Ready")

    def update_status(self, message):
        """Update status message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def on_closing(self):
        """Handle window closing"""
        if self.web_process:
            try:
                self.web_process.terminate()
            except:
                pass
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = BACCalculatorLauncher(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
