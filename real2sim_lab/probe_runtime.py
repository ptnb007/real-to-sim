import importlib.util
import json
import sys
import traceback
from pathlib import Path

root=Path(__file__).resolve().parent
repo=root.parent/'third_party/gs-real2sim'
sys.path.insert(0,str(repo/'submodule/AnySplat'))
modules=['torch','torchvision','einops','huggingface_hub','omegaconf','beartype','wandb','lightning',
         'torch_scatter','xformers','timm','e3nn','dacite','open3d']
report={'available_modules':{name:bool(importlib.util.find_spec(name)) for name in modules}}
try:
    from src.model.model.anysplat import AnySplat
    report['anysplat_import']='passed'
except Exception:
    report['anysplat_import']='failed'
    report['error']=traceback.format_exc()
(root/'runtime_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
