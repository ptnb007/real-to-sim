from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent / 'results/real_lab_mira/rgbd_pointcloud'
source = folder / 'white_divider.ply'
target = folder / 'white_divider_uncolored.ply'
with source.open('rb') as stream:
    header = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('Missing PLY header terminator')
        header.append(line.decode('ascii').strip())
        if header[-1] == 'end_header':
            break
    assert header[1] == 'format binary_little_endian 1.0'
    assert [s for s in header if s.startswith('property ')] == [
        'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue']
    count = int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
    dtype = np.dtype([(k, '<f4') for k in ('x', 'y', 'z')] + [(k, 'u1') for k in ('red', 'green', 'blue')])
    points = np.frombuffer(stream.read(), dtype=dtype)
    assert len(points) == count
xyz = np.column_stack([points[k] for k in ('x', 'y', 'z')]).astype('<f4')
new_header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {count}\nproperty float x\nproperty float y\nproperty float z\nend_header\n').encode('ascii')
with target.open('wb') as stream:
    stream.write(new_header)
    stream.write(xyz.tobytes())
with target.open('rb') as stream:
    assert stream.read(len(new_header)) == new_header
    restored = np.frombuffer(stream.read(), dtype='<f4').reshape(-1, 3)
assert np.array_equal(restored, xyz, equal_nan=True)
print(f'{target}\n{count} points; coordinates verified identical; no color properties.')
