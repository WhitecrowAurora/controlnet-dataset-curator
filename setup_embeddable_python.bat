@echo off
echo ========================================
echo Setup Embeddable Python
echo ========================================
echo.

cd /d H:\sucai\python

echo [1/3] Configuring Python path...
REM Edit python313._pth to enable site-packages
(
echo python313.zip
echo .
echo Lib
echo Lib\site-packages
echo.
echo import site
) > python313._pth

echo [Done]

echo.
echo [2/3] Installing pip...
REM Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

echo [Done]

echo.
echo [3/3] Verifying installation...
python -m pip --version

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Now you can install packages with:
echo   python -m pip install package_name
echo.
pause
