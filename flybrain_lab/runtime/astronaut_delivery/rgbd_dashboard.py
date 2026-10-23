"""Matplotlib evidence dashboard from measured RGB-D and executed robot logs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from pxr import Usd
from .simultaneous import TightBoundArm
from .robot_only_geometry import RobotOnlyGuard


def recorded_environment_depth(record,robot_guard):
    if 'robot_mask' in record:
        mask=np.asarray(record['robot_mask'],bool)
        if mask.shape!=record['depth'].shape:raise ValueError('recorded self mask shape mismatch')
        return np.where(mask,0,record['depth'])
    # Older recordings omitted the payload mask. Do not claim an exact replay.
    return robot_guard.mask_self_returns(record['depth'],record['K'],record['camera_world'],record['q'])


def make(run,scene,frames=64):
    root=Path(run);result=json.loads((root/'result.json').read_text());steps=json.loads((root/'steps.json').read_text())
    audit_path=root/'transfer_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    mission_path=root/'mission_audit.json'
    mission=json.loads(mission_path.read_text()) if mission_path.exists() else {}
    observations=json.loads((root/'observations.json').read_text());paths=json.loads((root/'paths.json').read_text())
    if not steps or not observations:raise ValueError('need actual executed motion and RGB-D observations')
    time=np.array([x['time_s'] for x in steps]);w=np.array([x['wrench'] for x in steps]);ot=np.array([x['time_s'] for x in observations])
    initial=np.load(root/'commissioning'/'measured_surface.npz')['vertices'];initial=initial[::max(1,len(initial)//12000)]
    stage=Usd.Stage.Open(str(scene));arms=[TightBoundArm(stage,s) for s in ('left','right')]
    robot_mask=RobotOnlyGuard(stage,*arms)
    fig=plt.figure(figsize=(14,11),layout='constrained');a=fig.add_subplot(321,projection='3d');b=fig.add_subplot(322)
    c=fig.add_subplot(323);d=fig.add_subplot(324);e=fig.add_subplot(325);f=fig.add_subplot(326)
    display_lower=np.array([-.1,-1,.4]);display_upper=np.array([1,.8,1.6])
    initial=initial[((initial>=display_lower)&(initial<=display_upper)).all(1)]
    a.scatter(*initial.T,s=.5,c='gray',alpha=.3);a.set(xlim=(-.1,1),ylim=(-1,.8),zlim=(.4,1.6),xlabel='X / m',ylabel='Y / m',zlabel='Z / m')
    a.view_init(elev=24,azim=-45);lines=[a.plot([],[],[],color=color,lw=3)[0] for color in ('tab:blue','tab:orange')]
    a.set_title('Initial RGB-D surface + current filtered depth + robot FK')
    cloud=a.scatter([],[],[],s=1,c='tab:cyan',alpha=.5)
    first=np.load(root/'raw'/observations[0]['file']);im=b.imshow(first['rgb']);b.axis('off')
    for axis,color in enumerate(('tab:red','tab:green','tab:blue')):
        c.plot(time,w[:,axis],color=color,label='F'+str(axis+1));d.plot(time,w[:,axis+3],color=color,label='T'+str(axis+1))
    c.set(xlabel='Simulation time / s',ylabel='Combined C-frame force / N');d.set(xlabel='Simulation time / s',ylabel='Combined C-frame torque / Nm')
    c.legend(ncol=3);d.legend(ncol=3);c.grid(alpha=.2);d.grid(alpha=.2)
    gaps=[];desired=[];ages=[];pixels=[]
    path_times=np.array([p['time_s'] for p in paths])
    for i,t in enumerate(time):
        row=observations[max(0,np.searchsorted(ot,t,side='right')-1)]
        path=paths[max(0,np.searchsorted(path_times,t,side='right')-1)]
        estimate=row['targets'].get(path.get('visual_target_name',row.get('active_visual_target','bracket')))
        q=np.array(steps[i]['q']).reshape(2,8)
        # Commanded Cartesian waypoint changes between approach/lift/place.
        # Never keep plotting the original cap as if it were still stationary.
        goal=np.array(path['target'])+[0,0,path['standoff_m']]
        sensing=any(key in path.get('phase','') for key in ('acquisition','recovery','handover','rollback'))
        gaps.append(np.nan if sensing else np.linalg.norm(arms[1].tcp(q[1])[:3,3]-goal)*1000)
        ages.append(max(0,t-estimate.get('stamp',row['time_s'])) if estimate else np.nan)
        pixels.append(estimate['pixels'] if estimate else 0)
    e.plot(time,gaps,label='TCP to phase goal (right motion only)');e.axhline(1.,ls='--',color='gray',label='Near-contact arrival tolerance')
    e.set(xlabel='Simulation time / s',ylabel='Distance / mm');e.legend(fontsize=8);e.grid(alpha=.2)
    f.plot(time,ages,color='tab:purple',label='Target measurement age');f.axhline(.25,color='r',ls='--',label='Motion freshness limit')
    f.set(xlabel='Simulation time / s',ylabel='Age / s');f.legend(loc='upper left',fontsize=8)
    support=f.twinx();support.plot(time,pixels,color='tab:olive',alpha=.6);support.set_ylabel('Measured target pixels',color='tab:olive')
    f.grid(alpha=.2);cursors=[ax.axvline(0,color='k',ls='--') for ax in (c,d,e,f)]
    text=fig.suptitle('')
    def update(index):
        t=float(index);i=min(np.searchsorted(time,t),len(steps)-1);q=np.array(steps[i]['q']).reshape(2,8)
        for arm,angles,line in zip(arms,q,lines):
            fk=arm.fk(angles);p=np.array([fk[n][:3,3] for n in arm.names]);line.set_data_3d(*p.T)
        j=max(0,np.searchsorted(ot,t,side='right')-1);row=observations[j];z=np.load(root/'raw'/row['file']);im.set_data(z['rgb'])
        depth=recorded_environment_depth(z,robot_mask)
        yy,xx=np.where(np.isfinite(depth)&(depth>0));stride=max(1,len(xx)//7000);yy,xx=yy[::stride],xx[::stride]
        axial=depth[yy,xx];K=z['K'];T=z['camera_world']
        points=np.c_[(xx-K[0,2])*axial/K[0,0],-(yy-K[1,2])*axial/K[1,1],-axial]@T[:3,:3].T+T[:3,3]
        points=points[((points>=display_lower)&(points<=display_upper)).all(1)]
        cloud._offsets3d=tuple(points.T)
        b.set_title(f'Actual RGB-D colour | exposure {ot[j]:.2f} s | {z["depth"].shape[1]} x {z["depth"].shape[0]}')
        for cursor in cursors:cursor.set_xdata([t,t])
        phase=steps[i]['phase']
        label=(f'{len(mission.get("audited_released_and_retreated_objects",[]))} / {mission.get("actual_scene_object_count","?")} object cycles | Whole scene: {mission.get("scene_all_objects_delivered_audited",False)}'
            if mission else f'Single release-and-withdrawal cycle: {audit.get("post_release_retreat_audited",False)}')
        text.set_text(f'RGB-D-only cooperative guidance | {t:.2f} s | {phase}\nFinal run audit: {label}')
        z.close();return lines+[im,text,cloud]+cursors
    animation=FuncAnimation(fig,update,frames=np.linspace(time[0],time[-1],frames),blit=False)
    animation.save(root/'rgbd_guidance.gif',writer=PillowWriter(fps=12))
    update(time[-1]);fig.savefig(root/'rgbd_guidance_summary.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--frames',type=int,default=64);a=p.parse_args();make(a.run,a.scene,a.frames)


if __name__=='__main__':main()
