# Isaac Sim lightweight setup

This setup targets the current workstation: Windows 10, RTX 5060 8 GB and a
Ryzen 7 5700X. Isaac Sim 5.1 is used because it supports Windows 10. It is an
archived release, and the GPU is below the compatibility checker's 10 GB
minimum for this release,
so the first validation deliberately uses one Jetbot, one ground plane, a
960x540 viewport and Raytraced Lighting.

`create_jetbot_scene.py` performs a real headless import from NVIDIA's official
Jetbot USD asset, advances physics, verifies the two-wheel articulation, saves
the stage and writes a JSON verification record. `open_jetbot_scene.bat` opens
the resulting stage in the Isaac Sim GUI.

The first validation passed after overriding the Python experience's default
Vulkan backend with D3D12 and disabling multi-GPU. The imported Jetbot reports
two degrees of freedom, physics advances successfully, and the Isaac camera
produces `jetbot_scene_preview.png`.

The first milestone is deliberately independent of ROS 2. After this scene is
stable, the next integration milestone is RGB-D sensor output and a velocity
command adapter for VLFM. Windows ROS integration should use native ROS 2 Jazzy
through Pixi; the current WSL environment is not the supported Windows URDF
import workflow in Isaac Sim 6.

That RGB-D/velocity milestone is now implemented as a localhost bridge. See
`VLFM_ISAAC_INTEGRATION.md`. Run VLFM and its visual services inside the existing
WSL environment, then launch `run_vlfm_jetbot.bat` on Windows. The first useful
demo still requires replacing the empty ground-plane scene with an indoor scene
containing a visible human model.
