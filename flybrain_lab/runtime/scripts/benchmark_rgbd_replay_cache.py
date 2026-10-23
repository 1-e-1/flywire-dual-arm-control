"""Controlled raw-scan replay timing and byte-exact map equivalence."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import classical_fusion.rgbd_recording as recording
from classical_fusion.planning_map import PlanningMapConfig
from astronaut_delivery.temporal_evidence_map import TemporalEvidenceTSDF


def run(reference,output):
    root=Path(reference);cfg=json.loads((root/'run_configuration.json').read_text())
    config=PlanningMapConfig(**json.loads((root/'local_map/measured_surface.json').read_text())['config'])
    rows=[];recording._decoded_replay_cache=None
    for name in ('cold_decoded','warm_decoded','authenticated_streaming'):
        if name=='authenticated_streaming':recording.REPLAY_CACHE_LIMIT_BYTES=0
        mapper=TemporalEvidenceTSDF(config,device='cuda:0');mapper.patch_radii=(4,6,8,10,12,16,24,32)
        torch.cuda.synchronize();start=time.perf_counter()
        info=recording.replay_initial_scan(mapper,cfg['initial_scan'],expected_scene_token=cfg['scene_token'],
            expected_calibration_id=cfg['calibration_id'],assume_unchanged_scene=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        tensors={k:getattr(mapper,k) for k in ('distance','weight','free_views','surface_veto','view_bits','cell_change_epoch')}
        tensors.update({f'view_word_{k}':v for k,v in mapper.extra_view_bits.items()})
        hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in tensors.items()}
        row=dict(mode=name,replay_wall_s=elapsed,frames=mapper.frames,cells=int(mapper.distance.numel()),
            independent_views=len(mapper.evidence_poses),replay=info,map_sha256=hashes)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='map_sha256'}),flush=True)
        del mapper,tensors;gc.collect();torch.cuda.empty_cache()
    out=dict(scope='Three controlled replays of the SAME original scan and fine ROI; no task execution or scene-state update',
        raw_reference=str(root),all_map_arrays_byte_identical=all(r['map_sha256']==rows[0]['map_sha256'] for r in rows),
        warm_speedup_over_authenticated_streaming=rows[2]['replay_wall_s']/rows[1]['replay_wall_s'],rows=rows,
        caveats=['Single-machine ordered microbenchmark, not whole-mission real-time evidence',
            'OS file caching/order and GPU warmup influence times; startup commissioning remains separately required'])
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
    if not out['all_map_arrays_byte_identical']:raise RuntimeError('Cached replay changed map evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):run(a.reference,a.output)
