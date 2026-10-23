"""Frozen paired study tables and figures, retaining every scene outcome."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .formal_statistics import summarize


def build(folder):
    root=Path(folder);protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'results.json').read_text());s=summarize(protocol,rows)
    if not s['final']:raise ValueError('Final report requires all frozen scene proposals to be accounted for')
    (root/'statistics.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    lines=['# Frozen paired mission results','',
        f"All {s['requested_scenes']} proposals are retained; admission is a pregrasp CAD/IK screen, not a guarantee of task feasibility.",
        'The evaluated frozen controller is NOT the later high-resolution/geometry-repair integration.',
        '', '| Method | Audited successes / executed | Exact 95% CI | Successes / all proposals | Outcomes |',
        '|---|---:|---|---:|---|']
    for m in ('pose','set'):
        x=s['methods'][m];ci=x['conditional_exact_95_ci']
        ci_text=f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else 'Not applicable'
        lines.append(f"| {m} | {x['successes']}/{x['admitted_executed']} | {ci_text} | {x['successes']}/{s['requested_scenes']} | {x['outcomes']} |")
    lines+=['',f"Admission counts: {s['admission_counts']}. Exact paired McNemar: {s['paired_success']}.",'',
        f"Secondary endpoints below are conditional on {len(s['common_success_seeds'])} common-success scenes.",
        'Scene bootstrap intervals are exploratory, without multiplicity correction; no failed trajectory is assigned a fabricated reaction score.',
        '', '| Endpoint | Pose mean | Set mean | Set minus pose | Bootstrap 95% CI |',
        '|---|---:|---:|---:|---|']
    for name,x in s['paired'].items():
        d=x['set_minus_pose']
        if d is None:
            lines.append(f'| {name} | N/A | N/A | N/A | No common successes |');continue
        ci=d['percentile_bootstrap_95_ci']
        lines.append(f"| {name} | {x['pose_mean']:.6g} | {x['set_mean']:.6g} | {d['mean_difference']:+.6g} | [{ci[0]:+.6g}, {ci[1]:+.6g}] |")
    lines+=['','The normalized wrench RMS endpoint was added during analysis and is explicitly post-hoc.','',
        '## Non-success and admission records','', '| Seed | Method | Status | Reason |','|---:|---|---|---|']
    for r in sorted(rows,key=lambda r:r['seed']):
        if not r.get('admitted'):
            lines.append(f"| {r['seed']} | admission | {r.get('admitted')} | See scene admission.json / retained result |")
        for m,x in r.get('methods',{}).items():
            if not x.get('success'):
                lines.append(f"| {r['seed']} | {m} | {x.get('status')} | {str(x.get('error') or 'See method log / independent audit').replace('|','/')} |")
    lines+=['','Wall times were obtained with concurrent workloads, not as a hard real-time benchmark. Python hash seed was not pinned in this primary batch. Source identities and all failed/timeout outcomes are retained.',
        'The successful diagnostic repeat of seed 2016 does not replace either primary timeout; its process also failed to exit after valid output and was explicitly terminated. The cause of the primary timeouts is unresolved.','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes.flat[0];labels=['pose','set'];counts=[s['methods'][m]['successes'] for m in labels]
    ax.bar(labels,np.array(counts)/s['requested_scenes'],color=['#6688bb','#22aa99'])
    ax.set(ylim=(0,1),ylabel=f"Audited success / all {s['requested_scenes']} proposals",title='Admission and failures remain in denominator')
    for i,m in enumerate(labels):
        ci=s['methods'][m]['all_proposal_exact_95_ci'];mean=counts[i]/s['requested_scenes']
        ax.errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='none',color='black',capsize=5)
    common=[r for r in rows if r['seed'] in s['common_success_seeds']]
    for ax,kind,title in zip(list(axes.flat)[1:],('force','torque','time'),('Force RMS (N)','Torque RMS (N m)','Mission time (s)')):
        extract=lambda r,m:r['methods'][m]['simulation_time_s'] if kind=='time' else r['methods'][m]['reaction'][kind]['rms']
        a=np.array([extract(r,'pose') for r in common]);b=np.array([extract(r,'set') for r in common])
        if not len(a):
            ax.text(.5,.5,'No common-success scenes',ha='center',transform=ax.transAxes);ax.set_title(title);continue
        ax.scatter(a,b,s=25,alpha=.75,color='#238c86');lo=min(a.min(),b.min());hi=max(a.max(),b.max())
        ax.plot([lo,hi],[lo,hi],'--',color='.5');ax.set(xlabel='Pose equality',ylabel='Visual set',title=title+'; common successes only')
    fig.suptitle('Frozen paired study: mixed outcomes must remain visible')
    fig.savefig(root/'paired_results.png',dpi=160);plt.close(fig)
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    s=build(args.run);print(json.dumps(dict(final=s['final'],methods=s['methods'],common=len(s['common_success_seeds'])),indent=2))


if __name__=='__main__':main()
