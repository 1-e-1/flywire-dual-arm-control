"""Independent observable-background ablation; original workcell is unchanged.

These are actual rendered cabin walls, NOT fabricated far-depth/free pixels.
The scene builder and sensor may see USD; the environment planner may not.
"""
from pathlib import Path
from pxr import Usd,UsdGeom
from .online_scene import ONLINE_SCENE
from .scene import ROOT,cube,material


def build():
    path=ROOT/'isaacsim_model/astronaut_delivery_rgbd_cabin.usda'
    if path.exists():raise FileExistsError('Preserve existing cabin ablation')
    stage=Usd.Stage.CreateNew(str(path));stage.GetRootLayer().subLayerPaths=[str(ONLINE_SCENE).replace('\\','/')]
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    paint=material(stage,'rgbd_cabin',(.42,.46,.50),.05)
    lo=(-.9,-1.15,-.05);hi=(1.5,1.15,2.5)
    center=[(a+b)/2 for a,b in zip(lo,hi)];extent=[b-a for a,b in zip(lo,hi)]
    for axis in range(3):
        for side,value in enumerate((lo[axis],hi[axis])):
            p=center.copy();size=extent.copy();p[axis]=value;size[axis]=.04
            cube(stage,f'/World/Station/Cabin/Wall_{axis}_{side}',p,size,paint)
    stage.GetRootLayer().Save();return path


if __name__=='__main__':print(build())
