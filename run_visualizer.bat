@echo off
set VENV_DIR=venv

if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Running main.py...
python main.py

pause
