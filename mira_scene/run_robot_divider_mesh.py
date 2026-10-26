"""Run only Mira SAM3D stage 2 for the robot/divider/table photo."""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ['MIRA_SAM3D_ROOT'] = '/home/woom/Mira-Scene/external/sam-3d-objects'
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

repo = Path('/home/woom/Mira-Scene')
sys.path.insert(0, str(repo / 'infer_scripts'))
import numpy as np
import torch
from hydra.utils import instantiate
from utils.sam3d_utils import CustomInferencePipelinePointMap, _sam3d_config_and_workspace, merge_mask_to_rgba


class Stage2OnlyPipeline(CustomInferencePipelinePointMap):
    def init_ss_generator(self, *args, **kwargs): return None
    def init_ss_decoder(self, *args, **kwargs): return None
    def init_ss_encoder(self, *args, **kwargs): return None
    def init_ss_condition_embedder(self, *args, **kwargs): return None
    def override_ss_generator_cfg_config(self, *args, **kwargs): return None
    def run(self, *args, **kwargs):
        raise RuntimeError('Stage-2-only runner; use run_stage2')
    def instantiate_and_load_from_pretrained(self, *args, device='cuda', **kwargs):
        with torch.device('cpu'):
            model = super().instantiate_and_load_from_pretrained(*args, device='cpu', **kwargs)
        return model.to(device)


root = Path('/mnt/c/Users/woom/Desktop/woom/mira_scene/results/real_lab_mira/cases/robot_divider_scene')
output = root / 'mesh/sam3d_stage2'
output.mkdir(parents=True, exist_ok=True)
report = root / 'mesh_status.json'
status = {'status': 'starting', 'method': 'Mira official SAM3D stage 2 from CCM voxel outputs', 'objects': []}
try:
    _, workspace, cfg = _sam3d_config_and_workspace(str(repo / 'checkpoints/sam-3d-objects/checkpoints/pipeline.yaml'))
    cfg.workspace_dir = str(workspace)
    cfg.compile_model = False
    cfg.depth_model = None
    cfg.layout_post_optimization_method = None
    cfg.layout_post_optimization_method_GS = None
    cfg._target_ = '__main__.Stage2OnlyPipeline'
    pipe = instantiate(cfg)
    spec = importlib.util.spec_from_file_location('mira_mesh', repo / 'infer_scripts/3_inference_mesh.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.load_ccm_outputs(str(root / 'CCM'))
    if data is None:
        raise RuntimeError(f'Missing CCM outputs under {root / "CCM"}')
    for i, voxels in enumerate(data['voxel_coords_list']):
        rgba = merge_mask_to_rgba(data['scene_image'], data['masks'][i] > 0)
        coords = np.concatenate([np.zeros((len(voxels), 1), dtype=np.int32), voxels], axis=1)
        with torch.inference_mode():
            result = pipe.run_stage2(image=rgba, coords=torch.from_numpy(coords).cuda(), seed=42,
                                     with_texture_baking=False, use_vertex_color=True)
        glb = result.get('glb')
        if glb is None:
            raise RuntimeError(f'SAM3D returned no mesh for object {i}')
        path = output / f'{i:03d}.glb'
        glb.export(str(path))
        record = {'index': i, 'path': str(path), 'appearance': 'vertex color'}
        status['objects'].append(record)
        status['status'] = 'running'
        report.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print('MIRA_MESH_WRITTEN', path, flush=True)
    status['status'] = 'complete'
except Exception as exc:
    status.update(status='failed', error=repr(exc))
    traceback.print_exc()
    raise
finally:
    report.write_text(json.dumps(status, indent=2), encoding='utf-8')
