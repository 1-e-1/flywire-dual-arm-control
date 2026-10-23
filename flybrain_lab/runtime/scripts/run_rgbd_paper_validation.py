"""Frozen holdout plus independent clocks, maps and failure-accounted report.

Runs serially. Never tunes, resamples failed scenes, edits the paper, or labels
the project complete. --postprocess-only resumes audits without rerunning data.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rgbd_paired_evaluation import evaluate,command,read
from astronaut_delivery.source_freeze import unchanged


def postprocess(folder,*,maps=True):
    folder=Path(folder).resolve();data=read(folder/'evaluation.json')
    if not data or data.get('finished_trials')!=data.get('planned_trials') or not data.get('source_unchanged'):
        raise ValueError('complete unchanged-source trial batch required before final reporting')
    if not unchanged(ROOT,read(folder/'source_manifest.json')):raise ValueError('postprocessing source differs from frozen batch')
    details=[]
    def save():
        (folder/'postprocess.json').write_text(json.dumps(dict(scope='Independent post-execution audits; completion of a batch does not imply task success or project completion',
            planned_trials=data['planned_trials'],maps_requested=maps,processed_trials=len(details),rows=details),indent=2),encoding='utf-8')
    def launch(module,args,stem):
        for i in range(32):
            log=folder/(stem+('' if i==0 else f'_retry_{i}')+'.log')
            if not log.exists():return command(module,args,log)
        raise RuntimeError('bounded audit retry log budget exhausted')
    save()
    for row in data['rows']:
        item=dict(seed=row['seed'],mode=row['mode'],recorded_clock=None,map_audits=[])
        result=read(row.get('result'));run=Path(row['result']).parent if row.get('result') else None
        if run is not None and all((run/n).is_file() for n in ('paths.json','steps.json','observations.json','run_configuration.json')):
            if not (run/'wall_clock_audit.json').exists():
                item['clock_process']=launch('scripts.audit_rgbd_wall_clock',['--run',run],f'clock_{row["seed"]}_{row["mode"]}')
            item['recorded_clock']=read(run/'wall_clock_audit.json')
        if maps and run is not None:
            cfg=read(run/'run_configuration.json');scene=cfg.get('scene')
            for label,where in [('broad',run),('local',run/'local_map')]:
                if not scene or not all((where/n).is_file() for n in ('planning_map_state.npz','measured_surface.npz','measured_surface.json')):continue
                for persistent in (False,True):
                    if persistent and not (where/'persistent_measured_surface.npz').is_file():continue
                    name='rgbd_static_interval_audit'+('_persistent' if persistent else '')+'.json'
                    record=dict(map=label,persistent=persistent,path=str(where/name))
                    if not (where/name).exists():
                        record['process']=launch('astronaut_delivery.rgbd_environment_audit',
                            ['--run',where,'--scene',scene,'--motion-run',run,'--static-interval-fusion',*(['--persistent-surface'] if persistent else [])],
                            f'map_{row["seed"]}_{row["mode"]}_{label}_{int(persistent)}')
                    record['report_present']=(where/name).is_file();item['map_audits'].append(record)
        details.append(item);save()
    statistics=folder/'statistics.json'
    if not statistics.exists():launch('scripts.report_rgbd_paired_evaluation',['--evaluation',folder/'evaluation.json','--output',statistics],'statistics')
    summary=dict(trials=data['finished_trials'],statistics_present=statistics.is_file(),
        recorded_clock_audits_passed=sum(d['recorded_clock'].get('passed') is True for d in details if d['recorded_clock']),
        recorded_clock_audits_available=sum(bool(d['recorded_clock']) for d in details),
        missing_map_reports=[a for d in details for a in d['map_audits'] if not a['report_present']],
        source_unchanged=unchanged(ROOT,read(folder/'source_manifest.json')),
        scope='Batch processing status, NOT all-task success, universal safety or completed research')
    (folder/'postprocess_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--postprocess-only',action='store_true')
    p.add_argument('--no-maps',action='store_true');p.add_argument('--protocol',default=str(ROOT/'configs/rgbd_holdout_protocol.json'));a=p.parse_args()
    if not a.postprocess_only:
        protocol=read(a.protocol)
        evaluate(a.output,protocol['seeds'],mission=True,extended_scan=True,modes=protocol['modes'],holdout_protocol=a.protocol)
    postprocess(a.output,maps=not a.no_maps)
