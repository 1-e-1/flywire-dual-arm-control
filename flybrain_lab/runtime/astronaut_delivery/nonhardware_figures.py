"""Audited accuracy/coverage and sensor trade-offs, including negative results."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    folders=['continuous_visual_nominal_01','high_resolution_4mm_01','postscan_coverage_clock_02']
    data=[json.loads((Path('reports')/f/'surface_accuracy_audit.json').read_text()) for f in folders]
    noise=json.loads(Path('reports/visual_error_holdout_20_01/visual_error_results.json').read_text())['summary']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colours=['#697a91','#227e85','#b48038']
    names=['12 mm\nmission','4 mm\nmission','4 mm + extra\npost-task scan']
    axes[0,0].bar(names,[d['measured_vertex_precision']['p95_m']*1000 for d in data],color=colours)
    axes[0,0].set(ylabel='Observed vertex error P95 / mm',title='Observed surface precision (lower is better)')
    axes[0,1].bar(names,[d['coverage_by_group']['parts']['within_5mm']*100 for d in data],color=colours)
    axes[0,1].set(ylabel='Complete part-surface coverage at 5 mm / %',ylim=(0,100),title='Unknown backsides remain missing')
    modes=list(noise);x=np.arange(len(modes))
    axes[1,0].bar(x,[100*noise[m]['detected_targets']/noise[m]['attempted_targets'] for m in modes],color='#227e85')
    axes[1,0].set(xticks=x,xticklabels=modes,ylabel='Detected target observations / %',ylim=(0,105),title='20 new scenes; repeated targets/views are clustered')
    axes[1,1].bar(x,[noise[m]['conditional_error_m']['p95']*1000 for m in modes],color='#b48038')
    axes[1,1].set(xticks=x,xticklabels=modes,ylabel='Accepted target error P95 / mm',title='Errors conditional on detection; exclusions shown at left')
    for ax in axes.ravel():ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Simulation audit: precision does not imply completeness or map-only planning',fontsize=13)
    fig.savefig(out/'accuracy_coverage_sensor.png',dpi=180);plt.close(fig)
    (out/'figure_sources.json').write_text(json.dumps(dict(surface_runs=folders,visual_run='visual_error_holdout_20_01',
        postscan_note='Additional observed views and extra time, not an equal-time comparison. Uses the corrected observation-clock scan; original diagnostic remains separate.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();build(args.output)
