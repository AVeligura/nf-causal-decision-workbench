@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title NF-Causal Decision Workbench - Windows tester build

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\build_windows_tester.ps1"
set "BUILD_EXIT=%ERRORLEVEL%"
echo.
if "%BUILD_EXIT%"=="0" (
    echo Готово. Результат находится в папке release.
) else (
    echo Сборка завершилась с ошибкой %BUILD_EXIT%.
)
echo.
pause
exit /b %BUILD_EXIT%
