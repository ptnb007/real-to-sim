from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
destination=root.parent/'Real2Sim_Demo_SAM3_Stage1.zip'
files=[root/'run_demo_segmentation.py',root/'sam3_windows_compat.py',root/'check_access.py']
files += [p for p in (root/'demo_segmentation').rglob('*') if p.is_file()]
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file,file.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(destination, destination.stat().st_size)
