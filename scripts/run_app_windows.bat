@echo off
setlocal
call "%~dp0..\INSTALL_AND_RUN.bat" %*
exit /b %ERRORLEVEL%
