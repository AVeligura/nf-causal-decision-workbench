@echo off
setlocal
set "NF_CAUSAL_REPAIR=1"
call "%~dp0INSTALL_AND_RUN.bat" %*
exit /b %ERRORLEVEL%
