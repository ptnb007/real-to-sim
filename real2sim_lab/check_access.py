"""Check public/gated model access without printing credentials or fetching weights."""
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

root=Path(__file__).resolve().parent
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ.pop(key,None)
token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
token_path=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token'
if not token and token_path.is_file():
    token=token_path.read_text().strip()
report={'credential_present':bool(token),'models':[]}
for model,file in [('facebook/sam3','sam3.pt'),('facebook/sam-3d-objects','checkpoints/pipeline.yaml'),('lhjiang/anysplat','config.json')]:
    entry={'model':model,'probe_file':file}
    headers={'Authorization':'Bearer '+token} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/models/'+model,headers=headers),timeout=20) as response:
            metadata=json.load(response)
        entry['gated']=metadata.get('gated',False)
        entry['files']=[x['rfilename'] for x in metadata.get('siblings',[])]
        with urllib.request.urlopen(urllib.request.Request('https://huggingface.co/'+model+'/resolve/main/'+file,headers=headers,method='HEAD'),timeout=20) as response:
            entry['http_status']=response.status
    except urllib.error.HTTPError as error:
        entry['http_status']=error.code
    except Exception as error:
        entry['error_type']=type(error).__name__
    report['models'].append(entry)
    print(json.dumps({k:v for k,v in entry.items() if k!='files'}),flush=True)
(root/'access_report.json').write_text(json.dumps(report,indent=2))
