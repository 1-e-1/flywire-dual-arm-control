"""Deterministic USD structure, asset integrity and exact FK tests."""
import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from build_astronaut_dual_arm import OUT, USD, digest, frame, world
from astronaut_kinematics import DualArmKinematics

def validate():
    s=Usd.Stage.Open(str(USD))
    s.SetEditTarget(s.GetSessionLayer())
    prims=list(s.Traverse())
    rigid=[p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    revolute=[p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
    fixed=[p for p in prims if p.IsA(UsdPhysics.FixedJoint)]
    assert len(rigid)==35 and len(revolute)==16 and len(fixed)==19
    assert all('lanzi' not in str(p.GetPath()).lower() for p in prims)
    assert not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims)
    assert all(UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() for p in rigid)
    assert UsdPhysics.Scene(s.GetPrimAtPath('/World/PhysicsScene')).GetGravityMagnitudeAttr().Get()==0
    manifest=json.loads((OUT/'assembly_build_report.json').read_text(encoding='utf-8'))['assets']
    assert all(digest(OUT/name)==data['sha256'] for name,data in manifest.items())
    unresolved=[]
    for p in prims:
        for a in p.GetAttributes():
            if a.GetTypeName()==Sdf.ValueTypeNames.Asset:
                v=a.Get()
                if v and v.path and not v.resolvedPath and not v.path.endswith('.mdl'):
                    unresolved.append(str(a.GetPath())+':'+v.path)
    assert not unresolved,unresolved
    robot=DualArmKinematics(s)
    camera=s.GetPrimAtPath('/World/LeftArm/b8/Camera')
    left=s.GetPrimAtPath('/World/LeftArm/b8')
    right=s.GetPrimAtPath('/World/RightArm/b8')
    magnet=s.GetPrimAtPath('/World/RightMagnet')
    body=s.GetPrimAtPath('/World/Astronaut/Body')
    c=s.GetPrimAtPath('/World/C_structure')
    camera_local=world(camera)*world(left).GetInverse()
    magnet_local=world(magnet)*world(right).GetInverse()
    locked=[np.array(world(body)),np.array(world(c))]
    max_joint_position=0.
    max_fixed_matrix=0.
    max_mount_slip=0.
    rng=np.random.default_rng(20260906)
    for i in range(128):
        q=rng.uniform(-.7,.7,(2,8))
        robot.set_positions(*q)
        for p,j,b0,b1,f0,f1 in robot.edges:
            w0=world(s.GetPrimAtPath(b0)) if b0 else Gf.Matrix4d(1)
            w1=world(s.GetPrimAtPath(b1))
            a,b=f0*w0,f1*w1
            max_joint_position=max(max_joint_position,(a.ExtractTranslation()-b.ExtractTranslation()).GetLength())
            if p.IsA(UsdPhysics.FixedJoint):
                max_fixed_matrix=max(max_fixed_matrix,float(np.max(np.abs(np.array(a)-np.array(b)))))
        max_mount_slip=max(max_mount_slip,float(np.max(np.abs(np.array(world(camera)*world(left).GetInverse())-np.array(camera_local)))),float(np.max(np.abs(np.array(world(magnet)*world(right).GetInverse())-np.array(magnet_local)))))
        assert np.allclose(world(body),locked[0],atol=1e-10)
        assert np.allclose(world(c),locked[1],atol=1e-10)
        for p in rigid:
            R=np.array(world(p))[:3,:3]
            assert np.allclose(R@R.T,np.eye(3),atol=2e-6),p.GetPath()
            assert np.linalg.det(R)>0
    assert max_joint_position<1e-6
    assert max_fixed_matrix<1e-6
    assert max_mount_slip<1e-6
    # Reusing the planner means matching the full moving chain, not just base XYZ.
    mirror_errors=[]
    mirror_axis_errors=[]
    for _ in range(128):
        q=rng.uniform(-1.2,1.2,8)
        robot.set_canonical_positions(q,q)
        for name in ['base/a1']+[n+str(i) for i in range(1,9) for n in ('a','b') if not(n=='a' and i==1)]:
            l=world(s.GetPrimAtPath('/World/LeftArm/'+name))
            r=world(s.GetPrimAtPath('/World/RightArm/'+name))
            mirror_errors.append(float(np.max(np.abs(robot.mirror_world_pose(l)-np.array(r)))))
        for i in range(1,9):
            name='base/a1' if i==1 else f'a{i}'
            axes=[]
            for side in ('LeftArm','RightArm'):
                joint=UsdPhysics.Joint(s.GetPrimAtPath(f'/World/{side}/{name}/joint{i}'))
                axes.append(np.array(frame(joint,0)*world(s.GetPrimAtPath(f'/World/{side}/{name}')))[2,:3])
            mirror_axis_errors.append(float(np.linalg.norm(robot.mirror_world_points(axes[0])-axes[1])))
    assert max(mirror_errors)<2e-6, max(mirror_errors)
    assert max(mirror_axis_errors)<2e-6,max(mirror_axis_errors)
    probes=rng.normal(size=(100,3))
    assert np.array_equal(robot.mirror_world_points(robot.mirror_world_points(probes)),probes)
    assert np.array_equal(robot.canonical_to_native('right',robot.canonical_to_native('right',q)),q)
    for bad in ([0]*7,[float('nan')]*8):
        try:
            robot.set_positions(bad,[0]*8)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    robot.set_positions([0]*8,[0]*8)
    before=robot.q.copy()
    robot.step_towards([1]*8,[-1]*8)
    assert np.max(np.abs(robot.q-before))<=.25/60+1e-12
    robot.set_positions([0]*8,[0]*8)
    bounds={}
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    for path in ('/World/Astronaut','/World/C_structure','/World/LeftArm','/World/RightArm','/World/RightMagnet'):
        b=cache.ComputeWorldBound(s.GetPrimAtPath(path)).ComputeAlignedRange()
        bounds[path]=[list(b.GetMin()),list(b.GetMax())]
    # Front/back is an observable geometry property, not a prim Euler label.
    cmesh=UsdGeom.Mesh(s.GetPrimAtPath('/World/C_structure/Geometry/mesh'))
    cv=np.asarray(cmesh.GetPointsAttr().Get())
    cw=(np.c_[cv,np.ones(len(cv))]@np.array(world(cmesh.GetPrim())))[:,:3]
    central=cw[np.abs(cw[:,1])<.1]
    assert central[:,0].min()>.2,'C middle crossbar is not in front of astronaut'
    assert np.allclose(np.array(world(c))[:3,:3][2],[0,0,1],atol=1e-6)
    assert not s.GetPrimAtPath('/World/Astronaut/Body/Collision').IsActive()
    pose_report=json.loads((OUT/'astronaut'/'work_pose_report.json').read_text(encoding='utf-8'))
    assert pose_report['all_visual_triangles_assigned']
    # AABB separation provides a conservative certified lower bound along Z
    # for the authored static human-arm convex hulls versus waist hardware.
    arm_bounds={}
    for side in ('left','right'):
        for part in ('upper_arm','forearm','glove'):
            name=side+'_'+part
            hull=s.GetPrimAtPath('/World/Astronaut/Body/WorkPoseCollision/'+name)
            cb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','guide']).ComputeWorldBound(hull).ComputeAlignedRange()
            arm_bounds[name]=[list(cb.GetMin()),list(cb.GetMax())]
    static_gap=min(b[0][2] for b in arm_bounds.values())-bounds['/World/C_structure'][1][2]
    assert static_gap>0,'Human arm hull overlaps C height band; inspect exact clearance'
    assert all(arm_bounds[side+'_glove'][0][0]>.33 for side in ('left','right'))
    report=dict(status='PASS',seed=20260906,samples=128,rigid_bodies=len(rigid),revolute_joints=len(revolute),fixed_joints=len(fixed),max_joint_anchor_error_m=max_joint_position,max_fixed_frame_matrix_error=max_fixed_matrix,max_camera_or_tool_mount_matrix_slip=max_mount_slip,unresolved_assets=unresolved,source_hashes_unchanged=True,bounds_m=bounds,collision_safety='NOT CERTIFIED: random samples test FK only, not collision-free motion or real flange fit.')
    report['sagittal_mirror']=dict(samples=128,max_link_pose_matrix_error=max(mirror_errors),max_axis_vector_error=max(mirror_axis_errors),canonical_to_native_right_sign=[-1]*8)
    report['work_pose']=dict(c_middle_crossbar_world_x_m=[float(central[:,0].min()),float(central[:,0].max())],human_arm_hulls_vs_C_vertical_gap_lower_bound_m=static_gap,arm_hull_bounds_m=arm_bounds,all_visual_triangles_covered=True)
    (OUT/'assembly_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    validate()
