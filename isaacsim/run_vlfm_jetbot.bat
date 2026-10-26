@echo off
setlocal
set "ISAACSIM_ROOT=D:\isaacsim-5.1.0"
set "VLFM_GOAL=person"
set "VLFM_SERVER=http://127.0.0.1:12190"
set "INTERNNAV_SERVER=http://127.0.0.1:8888"
set "INTERNNAV_ENABLED=1"
set "VLFM_ISAAC_SCENE=C:\Users\woom\Desktop\woom\isaacsim\jetbot_scene.usda"

if not exist "%ISAACSIM_ROOT%\python.bat" (
  echo Isaac Sim was not found at %ISAACSIM_ROOT%
  pause
  exit /b 1
)

call "%ISAACSIM_ROOT%\python.bat" "%~dp0run_vlfm_jetbot.py"
pause
