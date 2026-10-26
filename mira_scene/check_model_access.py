import json
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

api = HfApi()
for repo, filename in [('facebook/sam3', 'sam3.pt'), ('facebook/sam-3d-objects', 'checkpoints/pipeline.yaml'), ('Yang-Tian/Mira-Scene', 'pipeline/model_index.json')]:
    try:
        meta = get_hf_file_metadata(hf_hub_url(repo, filename), token=True)
        print(json.dumps({'repo': repo, 'access': True, 'file': filename, 'bytes': meta.size}), flush=True)
    except Exception as exc:
        print(json.dumps({'repo': repo, 'access': False, 'error_type': type(exc).__name__, 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}), flush=True)
