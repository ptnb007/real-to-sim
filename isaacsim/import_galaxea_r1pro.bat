@echo off
set "ISAACSIM_ROOT=D:\isaacsim-5.1.0"
set "SCRIPT=%~dp0import_galaxea_r1pro.py"
call "%ISAACSIM_ROOT%\python.bat" "%SCRIPT%"
if errorlevel 1 (
  echo Galaxea R1 Pro import failed. Review the output above.
  pause
  exit /b 1
)
echo Galaxea R1 Pro import passed.
pause

