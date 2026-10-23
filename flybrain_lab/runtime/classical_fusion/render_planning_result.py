"""Show actual native RGB, GT and reconstructed triangle geometry, without completion."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'classical_fusion/outputs/planning_map_v3_full_depth'
    with np.load(root/'data/planning_shapes_512_full_depth/ep_0000.npz') as source:
        rgb=source['rgb'][105]; gt=source['gt_object_points'].copy()
    with np.load(output/'sample_mesh.npz') as source:
        vertices,faces=source['vertices'],source['faces']
    report=json.loads((output/'evaluation.json').read_text())['summary']['tsdf_active']
    fig=plt.figure(figsize=(15,5.4))
    ax=fig.add_subplot(131); ax.imshow(rgb); ax.axis('off'); ax.set_title('Actual camera frame 105 | 512 x 512')
    for col,title in [(2,'Ground truth | evaluation only'),(3,'Measured mesh | unknown left open')]:
        ax=fig.add_subplot(1,3,col,projection='3d')
        if col==2:
            ax.scatter(*gt.T,c=gt[:,2],cmap='viridis',s=.5,alpha=.6)
        else:
            mesh=Poly3DCollection(vertices[faces],facecolor='#2995a1',edgecolor='#1c6470',linewidth=.08,alpha=.95)
            ax.add_collection3d(mesh)
            path_file=output/'example_sweep.json'
            if path_file.exists():
                sweep=json.loads(path_file.read_text())
                line=np.asarray([sweep['start_world_m'],sweep['end_world_m']])
                ax.plot(*line.T,color='#ef861c',linewidth=3,label='20 mm-radius tool sweep')
                ax.scatter(*line.T,color='#ef861c',s=15)
                ax.legend(loc='lower left',fontsize=7)
        ax.set(xlim=(.45,1.05),ylim=(-.65,.65),zlim=(.02,.75),xlabel='x / m',ylabel='y / m',zlabel='z / m')
        ax.set_box_aspect((.6,1.3,.73)); ax.view_init(23,-45); ax.set_title(title)
    fig.suptitle(f"Random mixed-shape native scene | 8 views x 5 frames | Surface MAE {report['surface_mae_m']*1000:.2f} mm\n"
                 f"P@5mm {report['precision_5mm']*100:.2f}% | R@10mm {report['recall_10mm']*100:.2f}% | Not a hardware planning approval",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(output/'planning_geometry_comparison.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
