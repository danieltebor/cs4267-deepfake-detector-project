@echo off
REM filepath: setup_env.bat

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
deactivate