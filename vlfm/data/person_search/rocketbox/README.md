# Textured adult avatar — Microsoft Rocketbox

Source: https://github.com/microsoft/Microsoft-Rocketbox

Pinned revision: `0943055db6ec570bcef9f2c8b41c9e5467c808f9`

Original folder: `Assets/Avatars/Adults/Male_Adult_01`.
The original FBX, preview PNG, three color TGA textures and upstream `LICENSE.md`
are preserved here. Copyright (c) 2020 Microsoft; MIT License. Keep the license
with copies of the original and converted asset.

`Male_Adult_01_standing.glb` is a derived static asset made by
`scripts/convert_rocketbox.py` with Blender 4.2.9 LTS:

- Bake the rig after lowering the upper arms into a relaxed standing pose.
- Retain the original mesh, UVs and face/body/hair color textures.
- Export embedded PNG textures and static geometry to Y-up glTF.
- Use alpha-cutout hair cards. No animation or skinning is required by Habitat.
- Normal/specular textures are not used in this flat-lit Habitat visualization.

This is a realistically textured virtual avatar, **not a photograph, a verified
scan of a real person, or a real trapped person**. Source geometry and texture
pixels are not AI-generated or replaced. Conversion metadata and SHA256 hashes
are in `conversion.json`. Habitat scales the instance to 1.7 m.

Rebuild using the locally cached portable Blender (CPU only):

```bash
/home/woom/.cache/vlfm-tools/blender-4.2.9-linux-x64/blender -b -t 2 \
  -P /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/convert_rocketbox.py
```

Original library citation:
Gonzalez-Franco et al. (2020), *The Rocketbox Library and the Utility of Freely
Available Rigged Avatars*, Frontiers in Virtual Reality.
https://doi.org/10.3389/frvir.2020.561558

## Additional static poses

`poses/` contains `Male_Adult_01_seated.glb` and `Male_Adult_01_lying.glb`,
derived from the same FBX with `convert_rocketbox.py --pose seated --preview`
or `--pose lying --preview`. Both represent a 1.7 m standing-height adult,
are ground-aligned, and include the original color textures. Preview images
are CPU renders of re-imported GLBs; their ground/reference seat is not exported.
The standing asset and completed navigation results are unchanged.

These pose assets have not been navigation-tested. The existing demo's
standing-height scaling must be adapted before loading seated or lying assets.
`scripts/verify_person_pose_assets.py` checks their static GLB structure,
embedded textures, hashes and approximate posed dimensions.
