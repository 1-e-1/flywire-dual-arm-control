"""Time-aligned measured dashboard. No final map shown in earlier frames."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pxr import Gf,Usd,UsdGeom
from .motion import Arm
from .vision import MeshCamera
from .run import DualArmKinematics

def build(folder,frames=140):
    folder=Path(folder);load=lambda n:json.loads((folder/n).read_text())
    result=load('result.json');trajectory=load('trajectory.json');steps=load('control_steps.json');obs=load('telemetry/observations.json')
    ts=np.array([s['time_s'] for s in steps]);W=np.array([s['wrench'] for s in steps]);fn=np.linalg.norm(W[:,:3],axis=1);tn=np.linalg.norm(W[:,3:],axis=1)
    stage=Usd.Stage.Open(result['scene_path']);stage.SetEditTarget(stage.GetSessionLayer());robot=DualArmKinematics(stage);right=Arm(stage,'right')
    view=np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.35,-2.45,1.95),Gf.Vec3d(.32,-.22,.95),Gf.Vec3d(0,0,1)).GetInverse()).T
    camera=MeshCamera(640,640,shaded=True);guidance=[];quality=[]
    for row in obs:
        target=row['estimates'].get(row['active_target']);valid=row['target_fresh'] and target is not None
        tcp=right.tcp(np.array(row['q'])[1])[:3,3]
        guidance.append(float(np.linalg.norm(tcp-target['position'])*1000) if valid else np.nan)
        quality.append(float(row['diagnostic_quality_sufficient']))
    ot=np.array([r['exposure_s'] for r in obs]);guidance=np.array(guidance)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    selection=np.unique(np.linspace(0,len(trajectory)-1,min(frames,len(trajectory))).astype(int));images=[];frame_index=[]
    background='#0e1723';foreground='#d8e4f0';cyan='#46d6dc';orange='#ffb568'
    for counter,index in enumerate(selection):
        record=trajectory[index];t=record['time_s'];robot.set_positions(*np.array(record['q']))
        for name,pose in record['parts'].items():UsdGeom.Xformable(stage.GetPrimAtPath('/World/Parts/'+name)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose))
        rendered=camera.capture(stage,view,t)
        fig=plt.figure(figsize=(16,10.5),dpi=100,facecolor=background);grid=fig.add_gridspec(5,3,width_ratios=[1.4,1,1],height_ratios=[1.6,.85,.85,.85,.8],hspace=.65,wspace=.3)
        axes=[fig.add_subplot(grid[:3,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[0,2],projection='3d'),fig.add_subplot(grid[1,1:]),fig.add_subplot(grid[2,1:]),fig.add_subplot(grid[3,1:]),fig.add_subplot(grid[3:,0]),fig.add_subplot(grid[4,1:])]
        scene_ax,rgb_ax,map_ax,fax,tax,gax,info,maxis=axes
        for ax in axes:
            ax.set_facecolor(background);ax.tick_params(colors=foreground,labelsize=8);ax.title.set_color(foreground)
            for spine in ax.spines.values():spine.set_color('#3e5369')
            ax.xaxis.label.set_color(foreground);ax.yaxis.label.set_color(foreground)
        scene_ax.imshow(np.clip(rendered.rgb,0,1));scene_ax.axis('off');scene_ax.set_title('双臂运动 / USD 几何回放',color=foreground)
        past=[r for r in obs if r['exposure_s']<=t+1e-8];snapshots=[r for r in past if r['frame_file']]
        latest=past[-1] if past else None;snapshot=snapshots[-1] if snapshots else None;points=np.empty((0,3))
        if snapshot:
            rgb_ax.imshow(Image.open(folder/'telemetry'/snapshot['frame_file']))
            points=np.load(folder/'telemetry'/snapshot['map_file'])['points']
            if len(points)>3500:points=points[np.linspace(0,len(points)-1,3500).astype(int)]
            if len(points):map_ax.scatter(*points.T,c=points[:,2],cmap='viridis',s=1.8,alpha=.8)
        snapshot_label=f"t={snapshot['exposure_s']:.2f} s" if snapshot else '尚无快照'
        rgb_ax.axis('off');rgb_ax.set_title(f'D435i 观测（全幅 / ROI）\n{snapshot_label}',color=foreground,fontsize=10)
        map_ax.set(xlim=(.25,1.),ylim=(-.72,.48),zlim=(.65,1.6));map_ax.view_init(30,-65)
        map_ax.set_title(f'已融合 TSDF；{snapshot_label}',color=foreground,fontsize=10)
        map_ax.set_xlabel('X / m');map_ax.set_ylabel('Y / m');map_ax.set_zlabel('Z / m',color=foreground);map_ax.tick_params(axis='z',colors=foreground)
        for axis in (map_ax.xaxis,map_ax.yaxis,map_ax.zaxis):axis.pane.set_facecolor(background)
        mask=ts<=t+1e-8;visible=np.flatnonzero(mask);last=visible[-1] if len(visible) else None
        for j,c in enumerate(['#ed788c','#64c5fb','#87d994']):fax.plot(ts[mask],W[mask,j],color=c,linewidth=.8,label=['Fx','Fy','Fz'][j])
        fax.plot(ts[mask],fn[mask],color=orange,linewidth=1.4,label='|F|');fax.axhline(.35,color='#8e9bac',linestyle='--',linewidth=.8,label='0.35 N 设计目标')
        fax.set(ylabel='反作用力 / N',title='C 形件合成惯性反作用（不含吸附接触冲击）',xlim=(0,ts[-1]),ylim=(-max(fn)*1.1,max(fn)*1.15))
        fax.legend(loc='upper right',ncol=5,fontsize=8,facecolor=background,labelcolor=foreground)
        tax.plot(ts[mask],tn[mask],color=cyan,linewidth=1.4,label='|τ|');tax.axhline(.15,color='#8e9bac',linestyle='--',linewidth=.8)
        tax.set(ylabel='反作用力矩 / N·m',title='合成力矩；虚线为采样设计目标，并非硬保证',xlim=(0,ts[-1]),ylim=(0,max(tn)*1.2))
        omask=ot<=t+1e-8;gax.plot(ot[omask],guidance[omask],color=orange,linewidth=1.2)
        gax.set(xlim=(0,ts[-1]),ylim=(0,max(120,float(np.nanpercentile(guidance,95))) if np.isfinite(guidance).any() else 120),xlabel='仿真时间 / s',ylabel='视觉目标距离 / mm',title='TCP 至视觉估计目标的距离（缺失观测不补零）')
        for ax in (fax,tax,gax):ax.axvline(t,color=foreground,alpha=.35,linewidth=.8);ax.grid(alpha=.15)
        maxis.step([r['exposure_s'] for r in snapshots],[r['map_points'] for r in snapshots],where='post',color=cyan,label='TSDF 表面点数')
        map_note='局部变更退回未知，其余证据保留' if result.get('map_policy')=='local' else '抓取/放置后全图重置'
        maxis.set(xlim=(0,ts[-1]),xlabel='仿真时间 / s',ylabel='融合表面点数',title='建图进度；'+map_note)
        maxis.grid(alpha=.15);quality_axis=maxis.twinx();quality_axis.plot(ot[omask],np.array(quality)[omask],color=orange,alpha=.35,linewidth=.8)
        quality_axis.set(ylim=(-.05,1.1),yticks=[0,1],ylabel='视觉诊断充分 0/1');quality_axis.tick_params(colors=foreground);quality_axis.yaxis.label.set_color(foreground)
        info.axis('off')
        if last is None:F=T=0.;cycle=0.
        else:F=fn[last];T=tn[last];cycle=1000*steps[last].get('cycle_wall_s',steps[last]['command_wall_s'])
        if latest:
            age=t-latest['exposure_s'];quality_text='满足诊断阈值' if latest['diagnostic_quality_sufficient'] else '不足 / 不适用'
            details=f"最近观测年龄：{age*1000:.0f} ms\n地图更新：{latest['map_updates']} 次；地图版本：{latest['epoch']}\n视觉质量：{quality_text}（非控制硬约束）"
            if latest['target_fresh']:
                details+=f"\n目标支持：{latest['pixels']} 点；估计 σ：{latest['position_sigma_max_m']*1000:.1f} mm"
        else:details='尚无观测；尚无地图'
        if last is not None:
            ends=ts[:last+1];starts=ends-np.array([s['dt_s'] for s in steps[:last+1]])
            overlap=np.maximum(0,np.minimum(ends,t)-np.maximum(starts,t-.25))
            impulse=float(np.linalg.norm(np.sum(W[:last+1,:3]*overlap[:,None],axis=0)))
        else:impulse=0.
        timing_note='异步流式周期，不含段间规划；非硬实时认证' if 'isolated_async' in result.get('validation_scope','') else '含同步感知/遥测；不作实时性能认证'
        info.text(0,1,f'|F| = {F:.3f} N    |τ| = {T:.3f} N·m\n0.25 s 线冲量模长：{impulse:.4f} N·s\n{details}\n该步计算：{cycle:.1f} ms\n（{timing_note}）\n\n地图是观测到的部分表面，\n未知区域不能当作自由空间。',va='top',color=foreground,fontsize=11,linespacing=1.65)
        fig.suptitle(f'双臂视觉引导与低反作用  |  t={t:.2f} s  |  {record["phase"]}',color=foreground,fontsize=17,y=.98)
        fig.text(.5,.015,'时间对齐日志回放，约 8× 加速；惯性反作用估计 ≠ 接触冲击；点云为真实融合快照。',ha='center',color='#95a9bd',fontsize=10)
        fig.subplots_adjust(top=.91,bottom=.08,left=.035,right=.975);fig.canvas.draw()
        im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());images.append(im);plt.close(fig)
        frame_index.append(dict(frame=counter,time_s=t,map_exposure_s=snapshot['exposure_s'] if snapshot else None,latest_observation_s=latest['exposure_s'] if latest else None))
        if counter in (0,len(selection)//2,len(selection)-1):im.save(folder/f'dashboard_{counter:03d}.png')
        if counter%20==0:print('dashboard_frame',counter,len(selection),flush=True)
    # Preserve non-uniform simulation intervals, not uniform index playback.
    frame_times=np.array([r['time_s'] for r in frame_index]);intervals=np.diff(frame_times)
    durations=[max(10,int(round(dt/8*1000/10))*10) for dt in np.r_[intervals,np.median(intervals) if len(intervals) else .8]]
    images[0].save(folder/'dashboard.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
    (folder/'dashboard_alignment.json').write_text(json.dumps(frame_index,indent=2),encoding='utf-8')
    print(folder/'dashboard.gif',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--frames',type=int,default=140);args=p.parse_args();build(args.run,args.frames)

if __name__=='__main__':main()
