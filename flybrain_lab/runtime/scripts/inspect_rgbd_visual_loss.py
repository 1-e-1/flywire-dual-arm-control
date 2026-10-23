"""Display recorded pre/post sensing frames without renderer or environment truth."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect(run):
    root=Path(run);obs=json.loads((root/'observations.json').read_text())
    path=json.loads((root/'paths.json').read_text())[-1]
    before=[o for o in obs if o['time_s']<=path['time_s']+1e-8 and o['integrated']][-1]
    selected=[before,obs[-1]];fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,row in enumerate(selected):
        with np.load(root/'raw'/row['file']) as z:
            depth=np.where(z['robot_mask'],np.nan,z['depth']) if 'robot_mask' in z else z['depth']
            ax[0,col].imshow(z['rgb']);ax[1,col].imshow(np.where(depth>0,depth,np.nan),vmin=.2,vmax=1.5,cmap='viridis')
        fresh=[k for k,v in row['targets'].items() if 0<=row['time_s']-v['stamp']<=.25]
        ax[0,col].set_title(f"{row['file']} / t={row['time_s']:.3f} s\nfresh targets: {', '.join(fresh) or 'NONE'}")
        ax[1,col].set_title('Measured depth after recorded robot/payload masking')
        for a in ax[:,col]:a.axis('off')
    fig.suptitle('Actual recorded RGB-D before / after failed sensing-only motion')
    output=root/'visual_loss_inspection.png';fig.savefig(output,dpi=140);plt.close(fig);print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);inspect(p.parse_args().run)
