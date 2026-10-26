@echo off
setlocal
if not defined ISAACSIM_ROOT set "ISAACSIM_ROOT=D:\isaacsim-5.1.0"
call "%ISAACSIM_ROOT%\python.bat" "%~dp0run_isaac_bridge.py" --smoke --output "%~dp0results"
exit /b %errorlevel%
