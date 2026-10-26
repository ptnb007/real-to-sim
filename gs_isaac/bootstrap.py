import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'runtime' / 'extensions'))
sys.path[:0] = [str(ROOT / 'prebuilt'), str(ROOT / 'deps'), str(ROOT.parent / 'third_party' / 'GaussianRenderer' / 'src')]
