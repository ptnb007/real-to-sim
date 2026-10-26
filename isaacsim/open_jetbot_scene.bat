@echo off
set "ISAACSIM_ROOT=D:\isaacsim-5.1.0"
set "SCENE=C:\Users\woom\Desktop\woom\isaacsim\jetbot_scene.usda"
if not exist "%ISAACSIM_ROOT%\isaac-sim.bat" (
  echo Isaac Sim was not found at %ISAACSIM_ROOT%
  pause
  exit /b 1
)
if not exist "%SCENE%" (
  echo The verified Jetbot scene has not been generated yet.
  pause
  exit /b 1
)
call "%ISAACSIM_ROOT%\isaac-sim.bat" "%SCENE%" --/app/vulkan=false --/renderer/multiGpu/enabled=false --/rtx-transient/dlssg/enabled=false --/app/renderer/resolution/width=960 --/app/renderer/resolution/height=540
