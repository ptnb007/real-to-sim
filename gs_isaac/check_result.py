"""Return a reliable exit code even if Kit shutdown resets its process status."""
import json
from pathlib import Path
import sys

report = json.loads((Path(__file__).parent/'results'/sys.argv[1]).read_text(encoding='utf-8'))
print(json.dumps({k:v for k,v in report.items() if k not in ('error','render_error')}, ensure_ascii=False))
if report.get('physics_and_sync') not in (None,'passed'):
    raise SystemExit(1)
raise SystemExit(0 if report['rendered'] else 2)
