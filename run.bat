@echo off
chcp 65001 >nul
echo ========================================
echo ControlNet GUI - Startup Script
echo ========================================

cd /d "%~dp0"

if not exist "python\python.exe" (
    echo [ERROR] Portable Python not found!
    echo Expected location: python\python.exe
    pause
    exit /b 1
)

echo Using portable Python
echo.

REM Check if this is first run (no marker file exists)
if not exist "python\.deps_installed" (
    echo.
    echo ========================================
    echo First-time setup - Installing dependencies
    echo ========================================
    echo This will download ~5-6GB of packages...
    echo.

    REM Install PyTorch with CUDA support
    echo [1/10] Installing PyTorch and torchvision...
    echo Trying CUDA 12.8 for RTX 30/40/50 series...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

    REM Check CUDA availability
    python\python.exe -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 goto try_cu121
    echo Successfully installed PyTorch with CUDA 12.8 support.
    goto torch_done

    :try_cu121
    echo CUDA 12.8 not available, trying CUDA 12.1...
    python\python.exe -m pip uninstall torch torchvision -y >nul 2>&1
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu121

    python\python.exe -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 goto cuda_failed
    echo Successfully installed PyTorch with CUDA 12.1 support.
    goto torch_done

    :cuda_failed
    echo [WARNING] CUDA not available. PyTorch will run on CPU only.

    :torch_done
    echo.
    echo [2/10] Installing ONNX Runtime GPU 1.24.2...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location onnxruntime-gpu==1.24.2

    echo.
    echo [3/10] Installing PyQt5 5.15.11...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location PyQt5==5.15.11

    echo.
    echo [4/10] Installing OpenCV 4.13.0.92...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location opencv-python==4.13.0.92

    echo.
    echo [5/10] Installing NumPy 2.4.2...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location numpy==2.4.2

    echo.
    echo [6/10] Installing SciPy 1.17.1...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location scipy==1.17.1

    echo.
    echo [7/10] Installing SymPy 1.14.0...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location sympy==1.14.0

    echo.
    echo [8/10] Installing PyArrow 23.0.1...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location pyarrow==23.0.1

    echo.
    echo [9/10] Installing Pandas 3.0.1...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location pandas==3.0.1

    echo.
    echo [10/10] Installing other dependencies...
    python\python.exe -m pip install --force-reinstall --no-cache-dir --no-warn-script-location pillow tqdm requests controlnet-aux timm datasets huggingface-hub

    echo.
    REM Create marker file to indicate dependencies are installed
    echo. > "python\.deps_installed"

    echo ========================================
    echo Installation complete!
    echo ========================================
    echo.
)

REM Set Qt plugin path for PyQt5
set "QT_PLUGIN_PATH=%~dp0python\Lib\site-packages\PyQt5\Qt5\plugins"

python\python.exe main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Program execution failed!
    echo Error code: %ERRORLEVEL%
    pause
)

pause
