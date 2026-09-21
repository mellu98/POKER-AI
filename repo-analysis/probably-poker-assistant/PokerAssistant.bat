@echo off
REM Poker Assistant Launcher
REM Double-click this file or create a shortcut to it on your desktop

cd /d "%~dp0"
call venv\Scripts\activate.bat
start "" pythonw launch_panel.pyw
