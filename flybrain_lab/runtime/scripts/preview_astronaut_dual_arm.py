"""Project actual USD triangles; this is NOT an Isaac Sim runtime render."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pxr import Usd, UsdGeom
from build_astronaut_dual_arm import USD, OUT

def preview():
    s=Usd.Stage.Open(str(USD))
    cache=UsdGeom.XformCache()
    tris=[]
    colors=[]
    for p in s.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue
        im=UsdGeom.Imageable(p)
        if im.ComputePurpose()=='guide' or im.ComputeVisibility()=='invisible' or 'collision' in str(p.GetPath()).lower():
            continue
        mesh=UsdGeom.Mesh(p)
        pts=np.asarray(mesh.GetPointsAttr().Get())
        pts=(np.c_[pts,np.ones(len(pts))]@np.array(cache.GetLocalToWorldTransform(p)))[:,:3]
        counts=np.array(mesh.GetFaceVertexCountsAttr().Get())
        indices=np.array(mesh.GetFaceVertexIndicesAttr().Get())
        faces=[]
        offset=0
        for n in counts:
            poly=indices[offset:offset+n]
            faces.extend([[poly[0],poly[i],poly[i+1]] for i in range(1,n-1)])
            offset+=n
        tri=pts[np.array(faces)]
        path=str(p.GetPath())
        color=('#cbd1d6' if '/Astronaut/' in path else '#168abb' if '/LeftArm/' in path else '#dd7739' if '/RightArm/' in path else '#d9ae28' if '/C_structure/' in path else '#73429a')
        from matplotlib.colors import to_rgb
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
        shading=.55+.45*np.abs(normals@np.array([.48,.36,.8]))
        colors.append(np.array(to_rgb(color))[None,:]*shading[:,None])
        tris.append(tri)
    tris=np.concatenate(tris)
    colors=np.concatenate(colors)
    fig,axes=plt.subplots(1,3,figsize=(16,8))
    for ax,(a,b),title in zip(axes,[(1,2),(0,2),(1,0)],['Front (looking from +X)','Side (looking from +Y)','Top (looking from +Z)']):
        d=({0,1,2}-{a,b}).pop()
        order=np.argsort(tris[:,:,d].mean(1))
        ax.add_collection(PolyCollection(tris[order][:,:,[a,b]],facecolors=colors[order],edgecolors='none',rasterized=True))
        ax.autoscale()
        ax.set(aspect='equal',title=title,xlabel='XYZ'[a]+' (m)',ylabel='XYZ'[b]+' (m)')
        ax.grid(alpha=.18)
        cam=s.GetPrimAtPath('/World/LeftArm/b8/Camera')
        mat=np.array(cache.GetLocalToWorldTransform(cam))
        pos=mat[3,:3]
        forward=-mat[2,:3]*.15
        ax.plot(pos[a],pos[b],'o',color='#007c63',markersize=5)
        ax.arrow(pos[a],pos[b],forward[a],forward[b],width=.006,color='#007c63',length_includes_head=True)
    fig.suptitle('Astronaut + CAD-referenced C mount + 2 x 8 DOF\nBlue: left camera arm | Orange: right tool arm | Yellow: C mount | Purple: magnet\nActual USD geometry projection (not runtime render)')
    fig.tight_layout()
    fig.savefig(OUT/'assembly_preview.png',dpi=150)
    print(OUT/'assembly_preview.png')

if __name__=='__main__':
    preview()
