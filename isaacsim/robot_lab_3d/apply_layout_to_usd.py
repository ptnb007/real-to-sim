#!/usr/bin/env python3
"""Apply the offline viewer's layout JSON to the supplied native USD scene.
Usage: python apply_layout_to_usd.py /path/to/lab_layout.json
No third-party dependencies. The original USD is never overwritten by default.
Viewer coordinates are glTF Y-up; USD is Z-up: (x, y, z) -> (x, -z, y).
"""
from pathlib import Path
import argparse, json, math, os, re

def apply_layout(layout_path: Path, source_path: Path, output_path: Path) -> None:
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    if layout.get('format') != 'lab_scene_layout_v1':
        raise ValueError('Expected a lab_scene_layout_v1 JSON exported by the supplied viewer.')
    scale = float(layout.get('scale', 1))
    if not math.isfinite(scale) or not 0 < scale < 10:
        raise ValueError('Scale must be finite, positive, and less than 10.')
    text = source_path.read_text(encoding='utf-8')
    text, count = re.subn(r'(double3 xformOp:scale = )\([^)]*\)',
                         lambda m: m.group(1) + f'({scale:.8f}, {scale:.8f}, {scale:.8f})',
                         text, count=1)
    if count != 1:
        raise ValueError('The source file is not the expected lab USD scene.')
    moved = 0
    for entry in layout.get('objects', []):
        identifier = entry.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_]+', identifier):
            raise ValueError(f'Invalid object identifier: {identifier!r}')
        delta = entry.get('translation', [0, 0, 0])
        if len(delta) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in delta):
            raise ValueError(f'Invalid translation for {identifier}.')
        x, y, z = delta
        xyz = f'({x:.8f}, {-z:.8f}, {y:.8f})'
        # Match only this named group's header, before any child mesh declarations.
        pattern = (r'(    def Xform "' + re.escape(identifier) + r'"\s*\{'
                   r'(?:(?!        def Mesh).)*?double3 xformOp:translate = )\([^)]*\)')
        text, count = re.subn(pattern, lambda m: m.group(1) + xyz, text, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'Object {identifier!r} was not found in the source USD.')
        moved += 1
    # Preserve texture resolution even when output is in another folder.
    texture_dir = source_path.parent / 'textures'
    relative_textures = Path(os.path.relpath(texture_dir, output_path.parent)).as_posix()
    text = text.replace('@textures/', '@' + relative_textures + '/')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')
    print(f'Wrote {output_path}\nUpdated {moved} object transforms; global scale {scale:.6f}.')
    print('Viewer hide/show flags are deliberately ignored; all source geometry is preserved.')
    print('This operation does not perform physics validation or metric calibration.')

def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', type=Path)
    parser.add_argument('--source', type=Path, default=root / 'robot_lab.usda')
    parser.add_argument('--output', type=Path, default=root / 'robot_lab_edited.usda')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Choose a different output filename to preserve the original USD.')
    try:
        apply_layout(args.layout, args.source, args.output)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.exit(1, f'Error: {error}\n')

if __name__ == '__main__':
    main()
