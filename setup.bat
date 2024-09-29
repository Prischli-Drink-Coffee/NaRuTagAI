@echo off

set PYTHON_VER=3.9

:: Проверка соотвествия версии python с рекомендуемой версией.
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Version python %PYTHON_VER% recommended.
)

IF NOT EXIST venv (
    echo creating venv...
    python -m venv venv
)

:: Создание папки logs.
mkdir ".\logs\setup" > nul 2>&1

:: Создание папки public.
mkdir ".\public" > nul 2>&1

:: Создание файла temp.
type nul > ".\temp.txt"

:: Деактивация активной среды.
call .\venv\Scripts\deactivate.bat

:: Вызов внешней программы Python для проверки локальных модулей.
python .\setup\check_local_modules.py

call .\venv\Scripts\activate.bat

REM Проверка, запуск двойным кликом .bat файла.
IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
    REM echo Скрипт был запущен двойным кликом.
    cmd /k python .\setup\setup_windows.py
) ELSE (
    REM echo Скрипт запущен камандной строкой.
    python .\setup\setup_windows.py
)

:: Деактивация вирутальной среды
call .\venv\Scripts\deactivate.bat
