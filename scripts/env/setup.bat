@echo off
REM Upgrade pip
python -m pip install --upgrade pip

REM Create a virtual environment
python -m venv .venv

REM Set CONAN_USER_HOME environment variable in the activate script
echo set CONAN_USER_HOME=%CD%>>.venv\Scripts\activate.bat

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install the required packages
pip install -r scripts/requirements.txt

REM Install pre-commit hooks
pre-commit install
