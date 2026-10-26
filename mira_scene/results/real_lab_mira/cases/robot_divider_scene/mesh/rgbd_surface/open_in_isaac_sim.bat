@echo off
call "D:\isaacsim-5.1.0\isaac-sim.bat" "%~dp0mira_rgbd_surface_mesh.usd" --/app/vulkan=false --/renderer/multiGpu/enabled=false --/rtx-transient/dlssg/enabled=false --/app/renderer/resolution/width=960 --/app/renderer/resolution/height=540
