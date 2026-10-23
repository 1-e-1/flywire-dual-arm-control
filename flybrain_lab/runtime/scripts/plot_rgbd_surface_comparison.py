"""Measured triangle visualization; no truth mesh, enclosure or hole filling."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def make(run):
    root=Path(run);reports=[json.loads((root/f'rgbd_environment_audit{s}.json').read_text()) for s in ('','_persistent')]
    lo=np.array([.20,-.62,.745]);hi=np.array([.47,-.09,.88])
    fig=plt.figure(figsize=(12,9),layout='constrained')
    for i,file in enumerate(('measured_surface.npz','persistent_measured_surface.npz')):
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        with np.load(root/file) as mesh:
            v=mesh['vertices'];triangles=v[mesh['triangles']]
        keep=((triangles>=lo)&(triangles<=hi)).all((1,2));triangles=triangles[keep]
        poly=Poly3DCollection(triangles,facecolor='#8fc9dc',edgecolor='#487e92',linewidth=.08,alpha=1.)
        ax.add_collection3d(poly);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X / m',ylabel='Y / m',zlabel='Z / m')
        ax.set_box_aspect(hi-lo);ax.view_init(elev=33,azim=-30)
        ax.set_title(('Current 8 mm TSDF only','Same map + retained measured triangles')[i],fontsize=11)
    ax=fig.add_subplot(223);names=list(reports[0]['per_object_coverage']);x=np.arange(len(names));width=.35
    for i,r in enumerate(reports):
        values=[r['per_object_coverage'][name]['coverage']['within_5mm']*100 for name in names]
        bars=ax.bar(x+(i-.5)*width,values,width,label=('Current TSDF','With measured memory')[i],color=('#abb8bf','#2989a9')[i])
        ax.bar_label(bars,fmt='%.1f',fontsize=9)
    ax.set(xticks=x,xticklabels=names,ylim=(0,100),ylabel='True surface within 5 mm / %',title='20,000 independent audit samples per object')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(224);ax.axis('off')
    a,b=reports;lines=['Same final scene state; identical free-space grid','',
        f'Aggregate 5 mm coverage: {a["dense_parts_coverage"]["within_5mm"]*100:.2f}% -> {b["dense_parts_coverage"]["within_5mm"]*100:.2f}%',
        f'Coverage distance P95: {a["dense_parts_coverage"]["p95_m"]*1000:.2f} -> {b["dense_parts_coverage"]["p95_m"]*1000:.2f} mm',
        f'Measured vertex P95: {a["surface_precision"]["p95_m"]*1000:.2f} -> {b["surface_precision"]["p95_m"]*1000:.2f} mm','',
        'Memory transports observed triangles only.', 'Unseen undersides remain unresolved.',
        'No extra free-space votes; no fabricated closure.', 'Top panels show measured surfaces, not CAD truth.',
        'Metric denominators cover the whole map, not this crop.']
    ax.text(.02,.95,'\n'.join(lines),va='top',fontsize=11,linespacing=1.65)
    fig.suptitle('RGB-D surface persistence after three measured transfers',fontsize=15)
    fig.savefig(root/'rgbd_surface_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();make(a.run)
