@echo off
rem QuickMIDI launcher -- double-click this file, or drag a .mid onto it.
rem A machine-specific override can live in run_local.bat (gitignored);
rem it receives all arguments and must launch qmidi itself.
setlocal
cd /d "%~dp0"
if exist "%~dp0run_local.bat" (
    call "%~dp0run_local.bat" %*
    goto :eof
)
set "PYTHONPATH=%~dp0src"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m qmidi %*
) else (
    python -m qmidi %*
)
if errorlevel 1 pause
