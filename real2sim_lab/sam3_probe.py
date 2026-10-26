"""Load the actual SAM3 image entry point using isolated dependencies."""
import sys
import traceback
from pathlib import Path

root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'deps'),str(root.parent/'third_party/sam3'),str(root.parent/'third_party/iopath')]
import sam3_windows_compat
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print('SAM3_IMPORT_PASSED')
except Exception:
    print(traceback.format_exc())
    raise
