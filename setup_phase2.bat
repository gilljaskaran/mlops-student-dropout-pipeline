@echo off
cd /d "%~dp0"
echo ============================== > setup_log.txt
echo Checking available Python versions >> setup_log.txt
py --list >> setup_log.txt 2>&1

echo ============================== >> setup_log.txt
echo Removing old .venv (was built with wrong Python version) >> setup_log.txt
if exist .venv (
    rmdir /s /q .venv >> setup_log.txt 2>&1
)

echo ============================== >> setup_log.txt
echo Creating new .venv with Python 3.11 >> setup_log.txt
py -3.11 -m venv .venv >> setup_log.txt 2>&1

echo ============================== >> setup_log.txt
echo Installing requirements >> setup_log.txt
call .venv\Scripts\python.exe -m pip install --upgrade pip >> setup_log.txt 2>&1
call .venv\Scripts\python.exe -m pip install -r requirements.txt >> setup_log.txt 2>&1
call .venv\Scripts\python.exe -m pip install -r requirements-api.txt >> setup_log.txt 2>&1
call .venv\Scripts\python.exe -m pip install -r requirements-monitoring.txt >> setup_log.txt 2>&1
call .venv\Scripts\python.exe -m pip install pytest flake8 black isort >> setup_log.txt 2>&1

echo ============================== >> setup_log.txt
echo DONE >> setup_log.txt
call .venv\Scripts\python.exe --version >> setup_log.txt 2>&1
call .venv\Scripts\python.exe -c "import pandas, sklearn, fastapi, dvc; print('all key packages import OK')" >> setup_log.txt 2>&1

echo.
echo Setup finished. See setup_log.txt for details.
echo.
pause
