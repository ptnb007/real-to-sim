from pathlib import Path
import numpy as np
from PIL import Image

source = Path(r'C:\Users\woom\Desktop\Mira_老师数据包\03_depth_raw_moge_v1.npy')
destination = Path(__file__).resolve().parent / 'relative_depth_raw_float32.tiff'
depth = np.load(source, allow_pickle=False)
assert depth.dtype == np.float32 and depth.ndim == 2
Image.fromarray(depth).save(destination, compression='tiff_deflate')
restored = np.asarray(Image.open(destination))
assert restored.dtype == depth.dtype
assert np.array_equal(depth, restored, equal_nan=True)
print(f'{destination}\n{depth.shape}: float32, exact round-trip verified; relative units, not meters')
