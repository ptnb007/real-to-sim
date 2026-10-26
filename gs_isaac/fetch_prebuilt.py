"""Install a checksum-pinned community build into an isolated local directory."""
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ.pop(key,None)
url = 'https://github.com/nagadomi/gsplat-windows-builds/releases/download/v20251230_6/gsplat-1.5.3%2Bpt27cu128-cp311-cp311-win_amd64.whl'
digest = '5ef27cff45308aeed1428e6117d2c7168b746056d098a91d522b18d78bfde086'
root = Path(__file__).parent
with urllib.request.urlopen(url) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == digest
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    archive.extractall(root/'prebuilt')
(root/'prebuilt_source.json').write_text(json.dumps({'url':url,'sha256':digest,'provider':'community Windows build; not official gsplat wheel'},indent=2))
print(f'Installed verified wheel: {len(data)} bytes')
