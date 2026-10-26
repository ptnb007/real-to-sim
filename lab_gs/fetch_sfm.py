import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ.pop(key,None)
with urllib.request.urlopen('https://pypi.org/pypi/pycolmap/3.13.0/json') as response:
    metadata=json.load(response)
wheel=next(x for x in metadata['urls'] if x['filename'].endswith('cp311-cp311-win_amd64.whl'))
with urllib.request.urlopen(wheel['url']) as response:
    content=response.read()
assert hashlib.sha256(content).hexdigest()==wheel['digests']['sha256']
with zipfile.ZipFile(io.BytesIO(content)) as archive:
    archive.extractall(Path(__file__).parent/'deps')
print(wheel['filename'])
