@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%deps;%PYTHONPATH%"
set "UV_CACHE_DIR=%ROOT%runtime\uv"
call D:\isaacsim-5.1.0\setup_python_env.bat
set "CARB_APP_PATH=D:\isaacsim-5.1.0\kit"
set "ISAAC_PATH=D:\isaacsim-5.1.0"
set "EXP_PATH=D:\isaacsim-5.1.0\apps"
"%ROOT%deps\uv-0.13.5.data\scripts\uv.exe" run --no-project --python D:\isaacsim-5.1.0\kit\python\kit.exe "%ROOT%render_views.py"
exit /b %errorlevel%
