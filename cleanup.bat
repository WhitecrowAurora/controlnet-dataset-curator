@echo off
echo ========================================
echo Cleanup for Distribution
echo ========================================
echo.

cd /d H:\sucai

echo [1/5] Deleting old script files...
del /q batch.py canny.py canny_rating.py cannyduqujiaoben.py cannyold.py 2>nul
del /q fullwithgui.py laplacianoperator.py local.py main_depth.py 2>nul
del /q rating.py steaming.py test.bat test_flow.py fix_torch.bat 2>nul
del /q BUGFIXES.md controlnet_gui_design.md 2>nul
echo [Done]

echo.
echo [2/5] Deleting cache folders...
rmdir /s /q __pycache__ 2>nul
rmdir /s /q .tmp 2>nul
rmdir /s /q temp 2>nul
echo [Done]

echo.
echo [3/5] Deleting test data...
rmdir /s /q extracted 2>nul
echo [Done]

echo.
echo [4/5] Optional cleanup...
echo Delete output folder? (Y/N, default N)
set /p "DEL_OUTPUT=: "
if /i "%DEL_OUTPUT%"=="Y" (
    rmdir /s /q output 2>nul
    echo [Deleted] output
) else (
    echo [Keep] output
)

echo.
echo Delete models folder? (Y/N, saves 1.3GB, default N)
set /p "DEL_MODELS=: "
if /i "%DEL_MODELS%"=="Y" (
    rmdir /s /q models 2>nul
    echo [Deleted] models
) else (
    echo [Keep] models
)

echo.
echo Delete python folder? (Y/N, users need to install Python, default N)
set /p "DEL_PYTHON=: "
if /i "%DEL_PYTHON%"=="Y" (
    rmdir /s /q python 2>nul
    echo [Deleted] python
) else (
    echo [Keep] python
)

echo.
echo [5/5] Cleaning config files...
del /q settings.json .progress.json 2>nul
echo [Done]

echo.
echo Delete large dependencies from site-packages? (Y/N, saves ~5GB, default Y)
echo (torch, onnxruntime, PyQt5, cv2, pyarrow, scipy, pandas - will auto-install on first run)
set /p "DEL_DEPS=: "
if /i "%DEL_DEPS%"=="" set "DEL_DEPS=Y"
if /i "%DEL_DEPS%"=="Y" (
    cd /d "%~dp0python\Lib\site-packages"

    REM Delete corrupted/temporary folders first
    echo Cleaning corrupted folders...
    rmdir /s /q ~orch ~-rch ~~rch ~.rch 2>nul
    rmdir /s /q ~umpy ~-mpy ~.mpy ~~mpy 2>nul
    rmdir /s /q ~umpy.libs ~-mpy.libs 2>nul

    REM Delete large dependencies
    echo Deleting large dependencies...
    rmdir /s /q torch torchvision torch-* torchvision-* 2>nul
    rmdir /s /q onnxruntime onnxruntime_gpu onnxruntime-* onnxruntime_gpu-* 2>nul
    rmdir /s /q PyQt5 PyQt5-* PyQt5_Qt5-* pyqt5_sip-* 2>nul
    rmdir /s /q cv2 opencv_python-* opencv_python_headless-* 2>nul
    rmdir /s /q pyarrow pyarrow-* pyarrow.libs 2>nul
    rmdir /s /q scipy scipy-* scipy.libs 2>nul
    rmdir /s /q pandas pandas-* 2>nul

    REM Delete marker file to trigger reinstall
    del /q .deps_installed 2>nul

    cd /d "%~dp0"
    echo [Deleted] Large dependencies (~5GB saved)
    echo [Note] Will auto-install on first run
) else (
    echo [Keep] Large dependencies
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
echo Package contents:
echo   - controlnet_gui\      (main code)
echo   - config.json          (config)
echo   - main.py              (entry point)
echo   - requirements.txt     (dependencies)
echo   - README.md            (documentation)
echo   - install.bat          (install script)
echo   - run.bat              (run script)
echo   - fix_pytorch.bat      (fix script)
echo   - clean_pytorch.bat    (clean script)
if exist python echo   - python\              (portable Python)
if exist models echo   - models\              (model files)
echo.
echo Options:
echo   1. Without python and models = Minimal (~50MB)
echo   2. With python, no models = Portable (~500MB)
echo   3. Full package = Complete (~6.5GB)
echo.
pause
