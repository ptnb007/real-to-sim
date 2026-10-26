"""Embed GLB and reference photographs in the offline viewer. Requires Pillow."""
from pathlib import Path
from io import BytesIO
import base64, json
from PIL import Image
root = Path(__file__).resolve().parents[1]
source = (root / 'source/viewer_template.html').read_text(encoding='utf-8')
refs = []
for path in sorted((root / 'references').glob('*.jpg')):
    image = Image.open(path).convert('RGB')
    image.thumbnail((1000, 750))
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=82)
    refs.append('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'))
source = source.replace('__MODEL_BASE64__', base64.b64encode((root / 'robot_lab.glb').read_bytes()).decode('ascii'))
source = source.replace('__REFERENCE_DATA__', json.dumps(refs))
(root / 'robot_lab_viewer.html').write_text(source, encoding='utf-8')
print('Wrote', root / 'robot_lab_viewer.html')
