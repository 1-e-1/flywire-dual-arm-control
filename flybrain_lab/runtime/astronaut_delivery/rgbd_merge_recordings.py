"""Reference immutable raw scans in one manifest without copying/re-timing data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def merge(paths,output):
    out=Path(output).resolve()
    if out.exists():raise ValueError('refuse to overwrite recording manifest')
    rows=[];sources=[];metadata=None;last=-np.inf
    for path in map(Path,paths):
        meta=json.loads(path.read_text(encoding='utf-8'))
        if meta.get('format')!='rgbd-initial-map-v1':raise ValueError('expected raw recording')
        if metadata is None:metadata=meta
        for key in ('scene_token','calibration_id','excluded_robot_hardware'):
            if metadata.get(key)!=meta.get(key):raise ValueError('incompatible recording/'+key)
        for row in meta['records']:
            raw=(path.resolve().parent/row['file']).resolve()
            if not raw.is_relative_to(path.resolve().parent) or not raw.is_relative_to(out.parent):raise ValueError('raw record escapes output root')
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('raw hash mismatch')
            with np.load(raw,allow_pickle=False) as data:stamp=float(data['exposure_s'])
            if not np.isfinite(stamp) or stamp<=last:raise ValueError('recording time overlap; raw timestamps must not be rewritten')
            last=stamp;rows.append(dict(row,file=raw.relative_to(out.parent).as_posix()))
        sources.append(dict(file=path.resolve().relative_to(out.parent).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows:raise ValueError('empty recording list')
    result=dict(format='rgbd-initial-map-v1',scene_token=metadata['scene_token'],calibration_id=metadata['calibration_id'],
        records=rows,source_manifests=sources,excluded_robot_hardware=metadata['excluded_robot_hardware'],
        acquisition='Combined immutable simulation raw RGB-D scans; external predeployment, not autonomous wrist motion',
        assumptions=metadata['assumptions'],environment_CAD_exported=False,renderer_instance_labels_exported=False)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=merge(a.inputs,a.output);print(json.dumps(dict(frames=len(result['records']),output=a.output)))
