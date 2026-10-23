"""Independent extra separating-axis search for conservative offline audits."""
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint,minimize
from scipy.spatial import ConvexHull


def certified_axis(va,vb,margin):
    """Optimizer proposes an axis; ALL vertices independently certify it.

    Face normals alone are incomplete for 3-D polyhedra (edge-edge separation).
    Solver success/tolerance never grants clearance; direct unit-axis projection
    must exceed the original margin. None means inconclusive, not collision.
    """
    va,vb=np.asarray(va,float),np.asarray(vb,float)
    if not np.isfinite(np.r_[va.ravel(),vb.ravel(),margin]).all() or margin<0:raise ValueError('invalid convex geometry')
    ea,eb=ConvexHull(va).equations,ConvexHull(vb).equations
    C=block_diag(ea[:,:3],eb[:,:3]);upper=-np.r_[ea[:,3],eb[:,3]]
    fun=lambda x:.5*np.dot(x[:3]-x[3:],x[:3]-x[3:])
    jac=lambda x:np.r_[x[:3]-x[3:],x[3:]-x[:3]]
    fit=minimize(fun,np.r_[va.mean(0),vb.mean(0)],jac=jac,
                 constraints=[LinearConstraint(C,-np.inf,upper)],method='SLSQP',
                 options=dict(ftol=1e-11,maxiter=60))
    direction=fit.x[:3]-fit.x[3:];length=np.linalg.norm(direction)
    if not np.isfinite(length) or length<1e-12:return None
    axis=direction/length;a,b=va@axis,vb@axis
    if max(a.min()-b.max(),b.min()-a.max())>margin+1e-9:return axis
    return None
