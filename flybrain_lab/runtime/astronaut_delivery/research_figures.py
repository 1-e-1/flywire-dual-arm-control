"""Paired effect sizes, preserving negative results and measured probe failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder);raw=json.loads((folder/'segments.json').read_text())
    rows=[r for r in raw if r['eligible'] and all(m['collision_certified'] for m in r['methods'].values())]
    if not rows:raise ValueError('No common collision-feasible cohort')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors={'frozen':'#697a91','pose':'#ec9d37','set':'#197c87'};names={'frozen':'Frozen camera','pose':'Full-pose constraint','set':'Visual set'}
    x=np.arange(len(rows))
    for metric,ax,unit in zip(('force','torque'),axes[:2],('N','N m')):
        for mode in colors:
            values=[r['methods'][mode]['reaction'][metric]['peak'] for r in rows]
            ax.plot(x,values,'o-',ms=3,lw=1,color=colors[mode],label=names[mode])
        ax.set(xlabel='Matched segment index',ylabel=f'{metric.title()} peak / {unit}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    effect=lambda r,metric:100*(r['methods']['set']['reaction'][metric]['peak']/r['methods']['frozen']['reaction'][metric]['peak']-1)
    for r in rows:
        sufficient=r['methods']['set']['all_measured_probes_sufficient']
        axes[2].scatter(effect(r,'force'),effect(r,'torque'),marker='o' if sufficient else 'x',color=colors['set'] if sufficient else '#b53945')
    axes[2].axvline(0,color='gray',lw=.8);axes[2].axhline(0,color='gray',lw=.8)
    axes[2].set(xlabel='Force-peak change vs frozen / %',ylabel='Torque-peak change vs frozen / %',title='Lower-left = both improve\nRed x = a measured visual probe failed')
    axes[2].grid(alpha=.2)
    fig.suptitle(f'Exploratory paired segments (n={len(rows)}): identical right trajectory and clock\nNot independent scenes, full-mission superiority, or continuous visibility evidence',fontsize=11)
    fig.savefig(folder/'comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();plot(args.run)
