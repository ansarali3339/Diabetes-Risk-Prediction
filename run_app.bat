@echo off
cd /d "%~dp0"

echo ==========================================
echo Diabetes Risk Prediction App
echo ==========================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo Installing/checking required packages...
%PYTHON_CMD% -m pip install -r requirements.txt

echo.
echo Starting Streamlit...
%PYTHON_CMD% -m streamlit run app.py

pause
