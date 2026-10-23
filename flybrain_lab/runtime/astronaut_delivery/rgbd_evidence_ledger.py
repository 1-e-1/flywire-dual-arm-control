"""Build an evidence index without promoting partial runs to full success."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path


def collect(reports):
    rows=[]
    for report in sorted(Path(reports).rglob('result.json')):
        folder=report.parent
        result=json.loads(report.read_text(encoding='utf-8'))
        if 'executed_segments' not in result:continue
        if result.get('environment_authority')!='RGBD_ONLY':continue
        audit_path=folder/'mission_audit.json'
        if not audit_path.exists():audit_path=folder/'transfer_audit.json'
        if not audit_path.exists():audit_path=folder/'execution_audit.json'
        audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
        events_path=folder/'actuation_events.json';events=json.loads(events_path.read_text()) if events_path.exists() else []
        capture=any(e['command']=='capture' and e['feedback']['attached'] for e in events)
        release=any(e['command']=='release' and not e['feedback']['attached'] for e in events)
        rows.append(dict(run=folder.relative_to(reports).as_posix(),status=result['status'],error=result.get('error'),segments=result['executed_segments'],
            ticks=result['executed_ticks'],simulation_time_s=result['simulation_time_s'],capture_confirmed=capture,release_confirmed=release,
            continuous_geometry_audited=audit.get('all_continuous_geometry_passed'),
            single_transfer_audited=audit.get('single_transfer_audited',False),
            retreat_audited=audit.get('post_release_retreat_audited',False),
            scene_all_objects_audited=audit.get('scene_all_objects_delivered_audited',False),
            completed_objects=audit.get('audited_released_and_retreated_objects',[]),
            full_transfer_audited=bool(result.get('full_transfer_completed') and audit.get('full_transfer_audited')),
            force_rms_N=(result.get('reaction') or {}).get('force',{}).get('rms'),
            torque_rms_Nm=(result.get('reaction') or {}).get('torque',{}).get('rms'),
            streaming_p99_ms=result.get('cycle_p99_ms')))
    return rows


def build(reports,output,tests=None):
    rows=collect(reports);test_result=None
    if tests:
        suite=ET.parse(tests).getroot().find('testsuite');test_result={k:suite.attrib[k] for k in ('tests','failures','errors','skipped','time')}
    data=dict(scope='Development ledger, not randomized success statistics; distinct algorithm versions and acquisition budgets',
        project_complete=False,remaining=['Reliable randomized multi-object measured delivery with retreat','New frozen paired randomized evaluation, holdout and ablations',
            'Whole-stack timing and online-control acceptance','Final simulation-only manuscript and reproducibility package'],
        hardware_in_scope=False,tests=test_result,runs=rows)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# Pure RGB-D development evidence ledger','',data['scope']+'.','',
        'Project complete: **No**. Hardware is excluded. A capture, release, independent audit, and complete multi-object mission are different endpoints.','',
        '| Run | Status | Segments | Capture | Release | Geometry | Single cycle | Audited completed objects | Whole scene |',
        '|---|---|---:|---|---|---|---|---|---|']
    lines += ['| '+ ' | '.join(str(row[k]) for k in ('run','status','segments','capture_confirmed','release_confirmed','continuous_geometry_audited','retreat_audited','completed_objects','scene_all_objects_audited'))+' |' for row in rows]
    lines+=['','Remaining:',*['- '+item for item in data['remaining']]]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(index=str(path),completed_runs=len(rows),tests=test_result)));return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports');p.add_argument('--output',required=True);p.add_argument('--tests');a=p.parse_args()
    build(a.reports,a.output,a.tests)
