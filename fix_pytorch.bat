@echo off
title Fix PyTorch Installation

echo ========================================
echo Fix PyTorch Installation
echo ========================================
echo.
echo Please close all Python programs first!
echo.
pause

echo.
echo Checking for embedded Python...
echo.

set "SCRIPT_DIR=%~dp0"
set "PYTHON_PATH=%SCRIPT_DIR%python\python.exe"
set "SITE_PACKAGES=%SCRIPT_DIR%python\Lib\site-packages"

if not exist "%PYTHON_PATH%" (
    echo [ERROR] Embedded Python not found!
    echo Looking for: %PYTHON_PATH%
    echo.
    pause
    exit /b 1
)

echo [Found] Python: %PYTHON_PATH%
echo [Found] site-packages: %SITE_PACKAGES%
echo.

echo Detecting CUDA version...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Found] NVIDIA GPU
) else (
    echo [Not Found] NVIDIA GPU
)

echo.
echo Select PyTorch version:
echo 1. CUDA 12.8 [DEFAULT]
echo 2. CUDA 12.6
echo 3. CUDA 12.4
echo 4. CPU version
echo.
set /p "CUDA_CHOICE=Enter (1-4, default 1): "

if "%CUDA_CHOICE%"=="" set "CUDA_CHOICE=1"
if "%CUDA_CHOICE%"=="1" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu128"
    set "CUDA_NAME=CUDA 12.8"
) else if "%CUDA_CHOICE%"=="2" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu126"
    set "CUDA_NAME=CUDA 12.6"
) else if "%CUDA_CHOICE%"=="3" (
    set "INDEX_URL=https://download.pytorch.org/whl/cu124"
    set "CUDA_NAME=CUDA 12.4"
) else if "%CUDA_CHOICE%"=="4" (
    set "INDEX_URL=https://download.pytorch.org/whl/cpu"
    set "CUDA_NAME=CPU"
) else (
    echo Invalid option
    pause
    exit /b 1
)

echo.
echo ========================================
echo Cleaning corrupted files
echo ========================================
echo.

cd /d "%SITE_PACKAGES%"

if exist torch (
    echo Deleting torch...
    rmdir /s /q torch 2>nul
    if exist torch (
        echo [ERROR] Cannot delete torch folder
        echo Please close all Python programs and try again
        pause
        exit /b 1
    )
    echo [Done] torch deleted
)

if exist torchvision (
    echo Deleting torchvision...
    rmdir /s /q torchvision 2>nul
    echo [Done] torchvision deleted
)

if exist ~orch (
    echo Deleting ~orch...
    rmdir /s /q ~orch 2>nul
)

if exist ~orchvision (
    echo Deleting ~orchvision...
    rmdir /s /q ~orchvision 2>nul
)

for /d %%i in (torch-*.dist-info) do (
    rmdir /s /q "%%i" 2>nul
)

for /d %%i in (torchvision-*.dist-info) do (
    rmdir /s /q "%%i" 2>nul
)

echo.
echo [Done] Cleanup complete
echo.

echo ========================================
echo Installing PyTorch (%CUDA_NAME%)
echo ========================================
echo.

"%PYTHON_PATH%" -m pip install torch torchvision --index-url %INDEX_URL%

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.

    echo Verifying...
    "%PYTHON_PATH%" -c "import torch; print('Version:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

    echo.
    echo Install depth-anything-v2? (Y/N)
    set /p "INSTALL_DEPTH=Enter (default N): "

    if /i "%INSTALL_DEPTH%"=="Y" (
        echo.
        echo Installing depth-anything-v2 (without reinstalling torch)...
        echo.

        REM Install depth-anything-v2 without dependencies first
        "%PYTHON_PATH%" -m pip install depth-anything-v2 --no-deps

        REM Then install other dependencies (excluding torch/torchvision)
        "%PYTHON_PATH%" -m pip install opencv-python numpy pillow huggingface-hub gradio gradio-imageslider

        echo.
        if %ERRORLEVEL% EQU 0 (
            echo [Done] depth-anything-v2 installed
        )
    )

    echo.
    echo All done! You can restart your program now.
) else (
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo Please check the errors above.
)

echo.
pause
