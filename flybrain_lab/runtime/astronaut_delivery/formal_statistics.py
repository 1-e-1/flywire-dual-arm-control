"""Scene-level, failure-preserving analysis of the frozen paired protocol."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import beta,binomtest


def interval(k,n,alpha=.05):
    if not n:return None
    return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def paired_difference(a,b,seed=9123,repeats=10000):
    """Difference b-a; resample SCENES, never control ticks or path segments."""
    delta=np.asarray(b,float)-np.asarray(a,float)
    if not len(delta):return None
    rng=np.random.default_rng(seed)
    means=delta[rng.integers(0,len(delta),(repeats,len(delta)))].mean(axis=1)
    return dict(n=len(delta),mean_difference=float(delta.mean()),
                percentile_bootstrap_95_ci=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_seed=seed,replicates=repeats)


def summarize(protocol,rows):
    expected=set(protocol['seeds']);by_seed={r['seed']:r for r in rows}
    if len(by_seed)!=len(rows) or not set(by_seed)<=expected:raise ValueError('duplicate or unexpected seeds')
    complete=len(rows)==len(expected)
    admitted=[r for r in rows if r.get('admitted') is True]
    attempted=[r for r in admitted if all(m in r.get('methods',{}) for m in ('pose','set'))]
    good=lambda r,m:bool(r.get('methods',{}).get(m,{}).get('success',False))
    result=dict(final=complete,requested_scenes=len(expected),recorded_scenes=len(rows),
                admission_counts=dict(Counter(str(r.get('admitted')) for r in rows)),
                complete_method_pairs=len(attempted),methods={},paired={},
                scope='Randomized three-interface task; common-success secondary endpoints are conditional, not unconditional superiority.',
                inference='Clopper-Pearson binomial intervals and paired scene bootstrap; secondary intervals are unadjusted exploratory intervals.')
    for method in ('pose','set'):
        k=sum(good(r,method) for r in admitted)
        n=sum(method in r.get('methods',{}) for r in admitted)
        result['methods'][method]=dict(successes=k,admitted_executed=n,
            conditional_success_fraction=k/n if n else None,conditional_exact_95_ci=interval(k,n),
            all_proposal_success_fraction=k/len(expected) if complete else None,
            all_proposal_exact_95_ci=interval(k,len(expected)) if complete else None,
            outcomes=dict(Counter(('AUDIT_FAILED' if r['methods'][method].get('status')=='PASS' and not good(r,method)
                                  else r['methods'][method].get('status','UNKNOWN')) for r in admitted if method in r.get('methods',{}))))
    wins=sum(good(r,'set') and not good(r,'pose') for r in attempted)
    losses=sum(good(r,'pose') and not good(r,'set') for r in attempted)
    result['paired_success']=dict(set_only=wins,pose_only=losses,
        exact_mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)
    common=[r for r in attempted if good(r,'pose') and good(r,'set')]
    result['common_success_seeds']=[r['seed'] for r in common]
    endpoints={'mission_time_s':lambda r:r['simulation_time_s']}
    for kind in ('force','torque'):
        for metric in ('peak','rms','maximum_window_impulse','accumulated_magnitude'):
            endpoints[kind+'_'+metric]=lambda r,k=kind,m=metric:r['reaction'][k][m]
    endpoints['normalized_wrench_rms_exploratory']=lambda r:float(np.hypot(r['reaction']['force']['rms']/.35,r['reaction']['torque']['rms']/.15))
    result['additional_exploratory_endpoint']='Normalized wrench RMS added during analysis, not a preregistered endpoint; force and torque remain separately reported.'
    for name,extract in endpoints.items():
        a=[extract(r['methods']['pose']) for r in common];b=[extract(r['methods']['set']) for r in common]
        result['paired'][name]=dict(pose_mean=float(np.mean(a)) if a else None,set_mean=float(np.mean(b)) if b else None,
                                   set_minus_pose=paired_difference(a,b))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    out=summarize(json.loads((root/'protocol.json').read_text()),json.loads((root/'results.json').read_text()))
    (root/'statistics.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
