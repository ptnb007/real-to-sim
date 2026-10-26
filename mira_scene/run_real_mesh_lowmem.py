"""Mira's real SAM3D stage2, excluding unused stage1 networks to save VRAM.

This does not replace model inference with procedural meshes. Only stage2 is
supported; stage1 calls are explicitly forbidden. Source checkpoints unchanged.
"""
import os, sys, json, traceback, importlib.util, argparse
from pathlib import Path
os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ['MIRA_SAM3D_ROOT'] = '/home/woom/Mira-Scene/external/sam-3d-objects'
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
repo = Path('/home/woom/Mira-Scene')
sys.path.insert(0, str(repo/'infer_scripts'))
sys.path.insert(0, str(repo/'external/MoGe'))
import numpy as np
import torch
from hydra.utils import instantiate
from utils.sam3d_utils import CustomInferencePipelinePointMap, _sam3d_config_and_workspace, merge_mask_to_rgba

class Stage2OnlyPipeline(CustomInferencePipelinePointMap):
    # None modules are accepted by ModuleDict. These networks are never used by
    # CustomInferencePipelinePointMap.run_stage2, which consumes Mira CCM voxels.
    def init_ss_generator(self, *args, **kwargs): return None
    def init_ss_decoder(self, *args, **kwargs): return None
    def init_ss_encoder(self, *args, **kwargs): return None
    # Stage 2 still uses the shared image/mask condition embedder.  Omitting it
    # makes the latent generator fall back to an identity lambda that cannot
    # accept the mask keyword produced by the official preprocessor.
    def init_ss_condition_embedder(self, *args, **kwargs):
        return super().init_ss_condition_embedder(*args, **kwargs)
    def override_ss_generator_cfg_config(self, *args, **kwargs): return None
    def run(self, *args, **kwargs):
        raise RuntimeError('Stage2-only runner; use Mira CCM input and run_stage2')
    def instantiate_and_load_from_pretrained(self, *args, device='cuda', **kwargs):
        # CPU construction prevents temporary CUDA copies while reading weights.
        with torch.device('cpu'):
            model = super().instantiate_and_load_from_pretrained(*args, device='cpu', **kwargs)
        return model.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/c/Users/woom/Desktop/woom/mira_scene/results/real_lab_mira')
    args = parser.parse_args()
    root = Path(args.root)
    status = {'status':'starting', 'method':'Mira CCM + official SAM3D stage2, unused stage1 omitted', 'objects':[]}
    report = root/'mesh_status.json'
    try:
        _, workspace, cfg = _sam3d_config_and_workspace(str(repo/'checkpoints/sam-3d-objects/checkpoints/pipeline.yaml'))
        cfg.workspace_dir = str(workspace)
        cfg.compile_model = False
        cfg.depth_model = None
        cfg.layout_post_optimization_method = None
        cfg.layout_post_optimization_method_GS = None
        cfg._target_ = '__main__.Stage2OnlyPipeline'
        pipe = instantiate(cfg)
        spec = importlib.util.spec_from_file_location('mira_mesh',repo/'infer_scripts/3_inference_mesh.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Smallest first for a genuine full-path smoke test.
        for case in sorted(path.name for path in (root/'cases').iterdir() if (path/'CCM').exists()):
            case_dir = root/'cases'/case
            data = module.load_ccm_outputs(str(case_dir/'CCM'))
            if data is None: raise RuntimeError(f'Missing CCM outputs: {case}')
            output = case_dir/'mesh/sam3d_stage2'
            output.mkdir(parents=True,exist_ok=True)
            for i, voxels in enumerate(data['voxel_coords_list']):
                rgba = merge_mask_to_rgba(data['scene_image'],data['masks'][i]>0)
                coords = np.concatenate([np.zeros((len(voxels),1),dtype=np.int32),voxels],axis=1)
                with torch.inference_mode():
                    result = pipe.run_stage2(image=rgba, coords=torch.from_numpy(coords).cuda(),seed=42,
                                             stage2_inference_steps=5,
                                             with_texture_baking=False, use_vertex_color=True)
                glb = result.get('glb')
                if glb is None: raise RuntimeError('SAM3D did not produce a mesh')
                path=output/f'{i:03d}.glb'
                glb.export(str(path))
                status['objects'].append({'case':case,'index':i,'path':str(path),'appearance':'vertex color; texture baking not yet run'})
                report.write_text(json.dumps(status,indent=2))
                print('MIRA_MESH_WRITTEN',path,flush=True)
        status['status']='object_meshes_complete_scene_alignment_pending'
    except Exception as exc:
        status.update(status='failed',error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        report.write_text(json.dumps(status,indent=2))
