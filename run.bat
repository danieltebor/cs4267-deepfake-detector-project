@echo off
REM filepath: run.bat

call venv\Scripts\activate.bat
python deepfake_detector.py
deactivate