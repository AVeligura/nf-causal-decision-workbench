@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title NF-Causal Decision Workbench - установка и запуск

set "PYTHON_EXE="
set "PYTHON_ARGS="

py -3.12 -c "import sys; raise SystemExit(0 if sys.maxsize ^> 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.12"
)

if not defined PYTHON_EXE (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize ^> 2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo.
    echo Не найден Python 3.12 x64.
    echo Для автоматической установки пакетов сначала требуется Python 3.12.
    echo Скачать: https://www.python.org/downloads/release/python-31210/
    echo При установке включите пункт "Add python.exe to PATH", затем запустите этот файл снова.
    echo.
    pause
    exit /b 2
)

echo.
echo NF-Causal Decision Workbench
echo Зависимости будут установлены в локальную папку .venv.
echo Системные пакеты Python не изменяются.
echo.

if /i "%NF_CAUSAL_REPAIR%"=="1" (
    "%PYTHON_EXE%" %PYTHON_ARGS% "scripts\bootstrap_windows.py" --repair --run -- %*
) else (
    "%PYTHON_EXE%" %PYTHON_ARGS% "scripts\bootstrap_windows.py" --run -- %*
)

set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Установка или запуск завершились с ошибкой %APP_EXIT%.
    echo Диагностика: installation.log
    echo.
    pause
)
exit /b %APP_EXIT%
