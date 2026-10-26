# Mira scoped lab reconstruction

Scope: the two divider boundaries, covered tables, and the floor between them. Robots and loose tabletop objects were excluded.

## Completed

- Robot-removed reference images: `cleaned/`
- SAM3 segmentation masks: `cases/*/input/`
- Mira CCM outputs: `cases/*/CCM/`
- Mira/MoGe depth and camera point maps: `cases/*/depth/moge/`
- Colored scoped surface meshes: `surface_mesh/*/mira_scoped_surface.{glb,ply,obj}`

The purple-divider view contains 234,858 vertices and 463,041 faces. The white-divider view contains 238,885 vertices and 469,486 faces.

## Important limitations

- These are single-view RGB-D surface meshes in each camera coordinate system. The two views are not registered into one shared metric frame.
- Depth scale is relative, not calibrated in metres.
- Areas revealed after removing robots are image inpainting, not measured geometry.
- Official SAM3D generative object-mesh decoding was attempted with 25 and 5 steps. On the RTX 5060 8 GB it remained at full GPU load without completing the first object, so those runs were stopped and are not presented as successful output.
