"""Render actual USD geometry and export kinematic trajectory for Isaac Sim replay."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pxr import Gf,Usd,UsdGeom
from .scene import ROOT,SCENE
from .vision import MeshCamera
from .run import DualArmKinematics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(ROOT/'reports/astronaut_delivery_verified'))
    parser.add_argument('--frames',type=int,default=100);args=parser.parse_args()
    folder=Path(args.run);folder.mkdir(parents=True,exist_ok=True)
    trajectory=folder/'trajectory.json'
    records=json.loads(trajectory.read_text()) if trajectory.exists() else []
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    scene_path=Path(result.get('scene_path',str(SCENE)))
    s=Usd.Stage.Open(str(scene_path));s.SetEditTarget(s.GetSessionLayer());robot=DualArmKinematics(s)
    camera=MeshCamera(960,720,shaded=True)
    view=np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.35,-2.45,1.95),Gf.Vec3d(.32,-.22,.95),Gf.Vec3d(0,0,1)).GetInverse()).T
    frames=[]
    selection=np.unique(np.linspace(0,len(records)-1,min(args.frames,len(records))).astype(int)) if records else [0]
    for j,index in enumerate(selection):
        row=records[index] if records else dict(time_s=0,phase='INITIAL WORKCELL')
        if records:
            robot.set_positions(*np.array(row['q']))
            for name,pose in row['parts'].items():
                UsdGeom.Xformable(s.GetPrimAtPath('/World/Parts/'+name)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose))
        obs=camera.capture(s,view,row['time_s'])
        im=Image.fromarray((obs.rgb*255).astype(np.uint8))
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,40),fill=(20,26,34))
        draw.text((12,8),f"ASTRONAUT DUAL-ARM | {row['time_s']:.2f} s | {row['phase']}",fill='white')
        draw.text((12,24),'USD geometry ray rendering / kinematic replay / not RTX',fill=(150,180,190))
        frames.append(im)
        if j==0:im.save(folder/'workcell_overview.png')
        if j%20==0:print('rendered',j,len(selection),flush=True)
    frames[-1].save(folder/'workcell_final.png')
    if len(frames)>1:
        durations=[max(20,int(1000*(records[b]['time_s']-records[a]['time_s']))) for a,b in zip(selection[:-1],selection[1:])]+[1000]
        frames[0].save(folder/'delivery_overview.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0)
    if records:
        replay='astronaut_delivery_cooperative_demo.usd' if result.get('controller')=='cooperative_certified_stream' else ('astronaut_delivery_online_demo.usd' if result.get('controller')=='joint_online_acceleration_qp' else ('astronaut_delivery_d435i_demo.usd' if result.get('sensor_profile') else 'astronaut_delivery_demo.usd'))
        output=ROOT/'isaacsim_model'/replay
        anim=Usd.Stage.CreateNew(str(output)) if not output.exists() else Usd.Stage.Open(str(output))
        anim.GetRootLayer().Clear();anim.GetRootLayer().subLayerPaths=[scene_path.name]
        # Stage-wide metrics are not inherited as ordinary prim properties from
        # a sublayer: explicitly preserve the source workcell's units/up axis.
        UsdGeom.SetStageMetersPerUnit(anim,UsdGeom.GetStageMetersPerUnit(s))
        UsdGeom.SetStageUpAxis(anim,UsdGeom.GetStageUpAxis(s))
        anim.SetDefaultPrim(anim.GetPrimAtPath('/World'));anim.SetTimeCodesPerSecond(60)
        anim.SetStartTimeCode(0);anim.SetEndTimeCode(records[-1]['time_s']*60)
        animated=DualArmKinematics(anim)
        # Samples are the logged FK states. Replay is visualization; execute joint
        # commands with run.py for the audited path, not interpolated USD matrices.
        for row in records:
            time=Usd.TimeCode(row['time_s']*60)
            animated.set_positions(*np.array(row['q']),time=time)
            for name,pose in row['parts'].items():
                UsdGeom.Xformable(anim.GetPrimAtPath('/World/Parts/'+name)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose),time)
        anim.GetRootLayer().Save();print(output,flush=True)

if __name__=='__main__':main()
