@echo off
set "ISAACSIM_ROOT=D:\isaacsim-5.1.0"
set "SCENE=%~dp0galaxea_r1pro_scene.usda"
if not exist "%SCENE%" (
  echo Galaxea R1 Pro scene has not been generated yet.
  echo Run import_galaxea_r1pro.bat first.
  pause
  exit /b 1
)
call "%ISAACSIM_ROOT%\isaac-sim.bat" "%SCENE%" --/app/vulkan=false --/renderer/multiGpu/enabled=false --/rtx-transient/dlssg/enabled=false --/app/renderer/resolution/width=960 --/app/renderer/resolution/height=540
if errorlevel 1 pause

