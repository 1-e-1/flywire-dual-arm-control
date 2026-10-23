"""Failure-accounted RGB-D paired pilot. Never imports CAD task controllers.

Execute from the project root. Each seed gets its own raw commissioning scan;
set and pose share that recording. The metadata is NOT sent to the controller.
The default is a single-object protocol; --mission requires all-object audits.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from astronaut_delivery.source_freeze import freeze,unchanged


def command(module,arguments,log):
    began=time.perf_counter()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
    with Path(log).open('x',encoding='utf-8') as stream:
        process=subprocess.run([sys.executable,'-m',module,*map(str,arguments)],cwd=ROOT,env=env,
            stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return dict(returncode=process.returncode,wall_s=time.perf_counter()-began,log=str(log))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).is_file() else {}


def reused_inputs(folder,seed,extended_scan):
    base=Path(folder)
    scene=base/f'scene_{seed}'/f'rgbd_random_{seed}.usda'
    recording=base/f'recording_{seed}.json' if extended_scan else base/f'scan_{seed}'/'recording.json'
    manifest=read(recording) if scene.is_file() else {}
    if manifest and len(manifest.get('records',[]))!=(732 if extended_scan else 516):
        raise ValueError('reused acquisition budget does not match protocol')
    return scene,recording,manifest


def evaluate(output,seeds,*,mission=False,extended_scan=False,modes=('set','pose'),reuse_inputs_from=None,holdout_protocol=None):
    if not modes or len(set(modes))!=len(modes) or not set(modes)<={'set','pose','hold'}:raise ValueError('invalid policy list')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('distinct nonempty scene seeds required')
    heldout=read(holdout_protocol) if holdout_protocol else None
    if holdout_protocol:
        if reuse_inputs_from is not None:raise ValueError('development recordings cannot be relabelled holdout')
        if (not heldout or heldout.get('seeds')!=list(seeds) or heldout.get('modes')!=list(modes)
            or not mission or not extended_scan or heldout.get('initial_frames')!=732):
            raise ValueError('holdout run must match the complete prespecified protocol')
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=seeds,modes=list(modes),selected_objects_per_trial='all observed' if mission else 1,initial_frames=732 if extended_scan else 516,
        paired_inputs='Same raw RGB-D scan; target auto selected from observed interfaces',
        split='prespecified_holdout' if heldout else 'development_pilot_not_holdout',scene_admission='none; all proposals including commissioning failures count',
        hardware=False,whole_mission=mission)
    if reuse_inputs_from is not None:protocol['reused_development_inputs_from']=str(reuse_inputs_from)
    if heldout:
        protocol['holdout_prespecification']=heldout
        protocol['holdout_protocol_sha256']=hashlib.sha256(Path(holdout_protocol).read_bytes()).hexdigest()
    hashes=freeze(ROOT,folder,protocol);(folder/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    def save():
        (folder/'evaluation.json').write_text(json.dumps(dict(protocol=protocol,source_unchanged=unchanged(ROOT,hashes),
            planned_trials=len(seeds)*len(modes),finished_trials=len(rows),rows=rows),indent=2),encoding='utf-8')
    save()
    for seed in seeds:
        if not unchanged(ROOT,hashes):raise RuntimeError('source changed; refusing mixed-version evaluation')
        scene_dir=folder/f'scene_{seed}';scene=scene_dir/f'rgbd_random_{seed}.usda';scan=folder/f'scan_{seed}'
        if reuse_inputs_from is None:
            generation=command('astronaut_delivery.rgbd_random_workcell',['--output',scene_dir,'--seed',seed],folder/f'generate_{seed}.log')
            acquisition=command('astronaut_delivery.rgbd_initial_scan',['--output',scan,'--scene',scene,'--dense-workspace'],folder/f'scan_{seed}.log') if generation['returncode']==0 else None
            recording=scan/'recording.json';manifest=read(recording)
        else:
            scene,recording,manifest=reused_inputs(reuse_inputs_from,seed,extended_scan)
            generation=dict(returncode=0 if scene.is_file() else 1,reused_scene=str(scene))
            acquisition=dict(returncode=0 if manifest else 1,reused_recording=str(recording),
                scope='Existing development measurements, not newly collected or held-out')
        extra=[]
        if manifest and extended_scan and reuse_inputs_from is None:
            corridor=folder/f'corridor_{seed}'
            extra.append(command('astronaut_delivery.rgbd_initial_scan',['--output',corridor,'--scene',scene,'--right-corridor-only','--start-time',30],folder/f'corridor_{seed}.log'))
            if extra[-1]['returncode']==0:
                merged=folder/f'recording_{seed}.json'
                extra.append(command('astronaut_delivery.rgbd_merge_recordings',['--inputs',recording,corridor/'recording.json','--output',merged],folder/f'merge_{seed}.log'))
                recording=merged;manifest=read(recording) if extra[-1]['returncode']==0 else {}
            else:manifest={}
        order=list(modes);offset=seed%len(order);order=order[offset:]+order[:offset]
        for mode in order:
            row=dict(seed=seed,mode=mode,generation=generation,acquisition=acquisition,single_transfer_audited=False,
                extra_acquisition=extra,retreat_audited=False,scene_all_objects_audited=False,status='COMMISSIONING_FAILED',result=None,audit=None)
            if manifest:
                run=folder/f'run_{seed}_{mode}'
                execution=command('astronaut_delivery.rgbd_mission' if mission else 'astronaut_delivery.rgbd_transfer',['--output',run,'--scene',scene,'--initial-scan',recording,
                    '--scene-token',manifest['scene_token'],'--calibration-id',manifest['calibration_id'],'--assume-unchanged-scene',
                    '--mode',mode,*([] if mission else ['--target','auto'])],folder/f'run_{seed}_{mode}.log')
                result=read(run/'result.json');row.update(execution=execution,result=str(run/'result.json'),status=result.get('status','PROCESS_FAILED'))
                if mission and (run/'paths.json').is_file():
                    audit_execution=command('astronaut_delivery.rgbd_mission_audit',['--run',run,'--scene',scene],folder/f'audit_{seed}_{mode}.log')
                    audited=read(run/'mission_audit.json');validated=audited.get('audited_released_and_retreated_objects',[])
                    row.update(audit=str(run/'mission_audit.json'),audit_execution=audit_execution,retreat_audited=bool(validated),
                        scene_all_objects_audited=bool(audited.get('scene_all_objects_delivered_audited')))
                elif any(e.get('event')=='measured_payload_attached' for e in result.get('transfer_events',[])):
                    audit_execution=command('astronaut_delivery.rgbd_transfer_audit',['--run',run,'--scene',scene],folder/f'audit_{seed}_{mode}.log')
                    audited=read(run/'transfer_audit.json');row.update(audit=str(run/'transfer_audit.json'),audit_execution=audit_execution,
                        single_transfer_audited=bool(audited.get('single_transfer_audited')),
                        retreat_audited=bool(audited.get('post_release_retreat_audited')))
                row['source_unchanged']=result.get('source_unchanged_during_run',False)
            rows.append(row);save();print(json.dumps(row),flush=True)
            if not unchanged(ROOT,hashes):raise RuntimeError('source changed during trial; exclude mixed-version evidence')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--mission',action='store_true');parser.add_argument('--extended-scan',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=['set','pose','hold'],default=['set','pose'])
    parser.add_argument('--reuse-inputs-from');parser.add_argument('--holdout-protocol')
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('duplicate seeds are not independent scenes')
    evaluate(args.output,args.seeds,mission=args.mission,extended_scan=args.extended_scan,modes=args.modes,
        reuse_inputs_from=args.reuse_inputs_from,holdout_protocol=args.holdout_protocol)
