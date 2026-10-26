import os, subprocess, json
from pathlib import Path
import psutil
repo = Path('/home/woom/Mira-Scene/external/pytorch3d')
out = Path('/mnt/c/Users/woom/Desktop/woom/mira_scene/results/real_lab_mira')
# Stop only our previously inspected redundant fallback build, not other tasks.
try:
    p = psutil.Process(611)
    cmd = ' '.join(p.cmdline())
    if 'pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/pytorch3d.git' in cmd:
        children = p.children(recursive=True)
        for child in reversed(children):
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        p.terminate()
        psutil.wait_procs(children+[p], timeout=5)
except psutil.NoSuchProcess:
    pass
if not repo.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/facebookresearch/pytorch3d.git',str(repo)],check=True)
prefix='/home/woom/miniconda3/envs/gsplat'
env=dict(os.environ)
env.update(CUDA_HOME=prefix, CPATH=prefix+'/targets/x86_64-linux/include',
           LIBRARY_PATH=prefix+'/targets/x86_64-linux/lib',
           CUB_HOME=prefix+'/targets/x86_64-linux/include',
           MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='12.0',
           PATH=prefix+'/bin:'+env['PATH'])
with (out/'pytorch3d_build.log').open('w') as log:
    r=subprocess.run(['/home/woom/mira_envs/ccm-rtx5060/bin/pip','install','-v',
                      '--no-build-isolation','--no-deps',str(repo)],env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'pytorch3d_build_status.json').write_text(json.dumps({'exit_code':r.returncode}))
print('PYTORCH3D_BUILD',r.returncode,flush=True)
