import sys
import os

os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='controlnet_aux')
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, module='timm')

torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib_path):
    print(f"Adding torch DLL path: {torch_lib_path}")
    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(torch_lib_path)
            print("Added torch DLL directory")
        except Exception as e:
            print(f"Warning: Could not add DLL directory: {e}")

print("Starting ControlNet GUI...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
try:
    os.makedirs(_tmp_dir, exist_ok=True)
    os.environ.setdefault("TEMP", _tmp_dir)
    os.environ.setdefault("TMP", _tmp_dir)
except Exception:
    pass


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401

        print(f"Preloaded torch: {torch.__version__}")
except Exception as e:
    print(f"[WARN] Torch preload failed (Depth/OpenPose may be unavailable): {e}")

print("\nChecking dependencies...")
missing_deps = []
optional_deps = []

required_deps = {
    'PyQt5': 'PyQt5',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'numpy': 'numpy',
}

optional_dep_map = {
    'controlnet_aux': 'controlnet-aux (for DWpose/Openpose)',
    'ultralytics': 'ultralytics (for ViTPose/SDPose-Wholebody)',
    'onnxruntime': 'onnxruntime-gpu (for ViTPose/SDPose-Wholebody)',
}

for module, package in required_deps.items():
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing_deps.append(package)

for module, desc in optional_dep_map.items():
    try:
        __import__(module)
        print(f"  ✓ {desc}")
    except ImportError:
        print(f" {desc} - Optional (some features unavailable)")
        optional_deps.append(desc)

if missing_deps:
    print(f"\n[ERROR] Missing required dependencies: {', '.join(missing_deps)}")
    print(f"Install with: pip install {' '.join(missing_deps)}")
    input("\nPress Enter to exit...")
    sys.exit(1)

if optional_deps:
    print(f"\n[INFO] Optional dependencies not installed:")
    for dep in optional_deps:
        print(f"  - {dep}")
    print("Some features may be unavailable. Continue anyway...")

print("\nAll required dependencies satisfied.")

print("Importing PyQt5...")
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

print("Importing main window...")
from controlnet_gui.gui.main_window import MainWindow


def main():
    print("Initializing Qt application...")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("ControlNet 控制图筛选工具")
    app.setOrganizationName("ControlNet")

    app.setStyle("Fusion")

    print("Creating main window...")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    window = MainWindow(config_path)

    def _bring_window_to_front():
        try:
            if window.isMinimized():
                window.showNormal()
            window.show()
            window.raise_()
            window.activateWindow()
            app.setActiveWindow(window)
        except Exception as e:
            print(f"[WARN] Failed to activate main window: {e}")

    print("Showing window...")
    _bring_window_to_front()
    QTimer.singleShot(0, _bring_window_to_front)
    QTimer.singleShot(250, _bring_window_to_front)
    QTimer.singleShot(800, _bring_window_to_front)

    print("Starting event loop...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
