"""Fetch pure Python packages and the uv launcher without changing Isaac's install."""
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
    os.environ.pop(key, None)
destination = Path(__file__).parent / 'deps'
destination.mkdir(exist_ok=True)
for package, version in [('gsplat', '1.5.3'), ('plyfile', '1.1.3'), ('uv', '0.13.5'), ('ninja', None), ('jaxtyping', None), ('wadler-lindig', None), ('rich', None), ('markdown-it-py', None), ('pygments', None), ('mdurl', None)]:
    endpoint = f'https://pypi.org/pypi/{package}/' + (f'{version}/' if version else '') + 'json'
    with urllib.request.urlopen(endpoint) as response:
        metadata = json.load(response)
    wheels = [item for item in metadata['urls'] if item['filename'].endswith('.whl') and
              ('py3-none-any' in item['filename'] or 'none-win_amd64' in item['filename'])]
    if not wheels:
        raise RuntimeError(f'No compatible wheel: {package}')
    with urllib.request.urlopen(wheels[0]['url']) as response:
        data = response.read()
    import hashlib
    assert hashlib.sha256(data).hexdigest() == wheels[0]['digests']['sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).exists():
                archive.extract(member, destination)
    print(wheels[0]['filename'], flush=True)
