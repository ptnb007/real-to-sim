@echo off
call "D:\isaacsim-5.1.0\python.bat" "%~dp0run_r1pro_grasp_demo.py" --record-data
if errorlevel 1 goto :failed
call "D:\isaacsim-5.1.0\python.bat" "%~dp0mcap_to_hdf5.py"
if errorlevel 1 goto :failed
echo Data pipeline complete.
pause
exit /b 0
:failed
echo Data pipeline failed. Check the console output.
pause
exit /b 1
