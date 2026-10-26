"""Preserve the nine supplied photos and record the user's reconstruction scope."""
from pathlib import Path
import shutil
import json

ROOT = Path(__file__).resolve().parent / 'results' / 'lab_between_dividers'
SOURCE = Path('C:/Users/woom/Documents/xwechat_files/wxid_3fg5lyf1apf612_e732/temp/RWTemp/2026-09/b69eae616f02c776265a0b6423a5cd6c')
NAMES = ['95c4b8ccedec90a0b64aaa6e4ee39327','b836ce3d834f8dce16f34a66658350a5','bf7d9c37fb3bdbe65152dd905dbb10f2','b72a0e426a713817ab75bbcba8ff06c0','2c0ff0f05074c56a75c16f7fb433fa9d','38a447b4732866e31ba0569052315b32','43494f12a82a6f60a899fdec7d6aedec','d0dd85f3a74afc738b964fd3b7f7df54','5473f5c5d4281be7c15cc84b828c47d6']
for folder in ['originals', 'cleaned', 'cases', 'logs']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
records = []
for i, name in enumerate(NAMES, 1):
    src = SOURCE / (name + '.jpg')
    dst = ROOT / 'originals' / f'{i:02d}_{name}.jpg'
    if not dst.exists():
        shutil.copy2(src, dst)
    records.append({'index': i, 'source': str(src), 'preserved_original': str(dst),
                    'edit_needed': i in [1, 2, 3, 7, 8],
                    'role': 'scope_reference_only' if i == 4 else 'candidate_reconstruction_view'})
(ROOT / 'input_manifest.json').write_text(json.dumps({
    'scope': 'Area between the two dividers: tables, tablecloths, and intervening floor',
    'exclude': ['all robots', 'loose objects on the foreground table', 'outside area'],
    'originals_preserved': True,
    'geometry_caveat': 'Image inpainting is hypothetical; missing observations are not measurements. Mira is a per-image pipeline; cross-view registration is separate.',
    'photos': records,
    'status': 'inputs_preserved_image_cleanup_in_progress'
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(ROOT)
