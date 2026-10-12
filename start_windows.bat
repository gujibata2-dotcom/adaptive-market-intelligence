@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
pause
exit /b
:fail
echo Setup failed. Install Python 3.11 or 3.12 and enable Add Python to PATH.
pause
exit /b 1
