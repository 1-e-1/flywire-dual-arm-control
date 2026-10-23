"""Bounded instantaneous reaction feasibility with a dual residual lower bound.

This is a diagnostic, not a replacement controller or a trajectory feasibility
claim. A numerically feasible zero is not a global reactionless motion proof.
"""
import numpy as np
from scipy.optimize import linprog,minimize,LinearConstraint


def bounded_residual(B,b,lower,upper,E=None,f=None,C=None,h=None):
    B,b,lower,upper=map(lambda x:np.asarray(x,float),(B,b,lower,upper))
    if B.ndim!=2 or min(B.shape)==0:raise ValueError('expected nonempty reaction matrix [wrench, joints]')
    n=B.shape[1]
    if B.ndim!=2 or b.shape!=(B.shape[0],) or lower.shape!=(n,) or upper.shape!=(n,) or np.any(lower>upper):raise ValueError('invalid dimensions or bounds')
    if not np.isfinite(np.r_[B.ravel(),b,lower,upper]).all():raise ValueError('finite bounds required')
    E=np.empty((0,n)) if E is None else np.asarray(E,float);f=np.empty(0) if f is None else np.asarray(f,float)
    C=np.empty((0,n)) if C is None else np.asarray(C,float);h=np.empty(0) if h is None else np.asarray(h,float)
    if E.shape!=(len(f),n) or C.shape!=(len(h),n) or not np.isfinite(np.r_[E.ravel(),f,C.ravel(),h]).all():raise ValueError('invalid linear constraints')
    bounds=list(zip(lower,upper));kwargs=dict(bounds=bounds,A_ub=C if len(C) else None,b_ub=h if len(C) else None)
    start=linprog(np.zeros(n),A_eq=E if len(E) else None,b_eq=f if len(E) else None,method='highs',**kwargs)
    if not start.success:return dict(status='TASK_SET_INFEASIBLE_OR_LP_INCONCLUSIVE',zero_feasible=False,residual_lower_bound=None)
    zero=linprog(np.zeros(n),A_eq=np.vstack([E,B]),b_eq=np.r_[f,-b],method='highs',**kwargs)
    if zero.success:
        x=zero.x
        violation=max(0,float(np.max(np.r_[lower-x,x-upper,E@x-f,f-E@x,C@x-h])))
        if np.linalg.norm(B@x+b)<1e-7 and violation<1e-8:
            return dict(status='NUMERICALLY_ZERO_FEASIBLE',zero_feasible=True,solution=x.tolist(),residual=float(np.linalg.norm(B@x+b)),residual_lower_bound=0.,constraint_violation=violation)
    constraints=[]
    if len(E):constraints.append(LinearConstraint(E,f,f))
    if len(C):constraints.append(LinearConstraint(C,-np.inf,h))
    fun=lambda x:.5*np.linalg.norm(B@x+b)**2
    fit=minimize(fun,start.x,jac=lambda x:B.T@(B@x+b),method='SLSQP',bounds=bounds,constraints=constraints,
                 options=dict(ftol=1e-12,maxiter=200))
    x=fit.x;residual=B@x+b;norm=np.linalg.norm(residual)
    direction=residual/max(norm,1e-15);objective=B.T@direction
    support=linprog(objective,A_eq=E if len(E) else None,b_eq=f if len(E) else None,method='highs',**kwargs)
    lower_bound=0.
    if support.success:
        # Any nonnegative lambda and free nu give a valid dual bound. Explicit
        # minimization over the box absorbs stationarity/numerical residuals.
        lam=np.maximum(0,-support.ineqlin.marginals) if len(C) else np.empty(0)
        nu=-support.eqlin.marginals if len(E) else np.empty(0)
        station=objective+C.T@lam+E.T@nu
        dual=-lam@h-nu@f+np.minimum(station*lower,station*upper).sum()+direction@b
        lower_bound=max(0.,float(dual)-1e-9)
    violation=max(0,float(np.max(np.r_[lower-x,x-upper,E@x-f,f-E@x,C@x-h])))
    return dict(status='POSITIVE_RESIDUAL_CERTIFIED' if lower_bound>1e-7 else 'ZERO_NOT_ESTABLISHED',zero_feasible=False,
                solution=x.tolist(),residual=float(norm),residual_lower_bound=lower_bound,constraint_violation=violation,
                direction=direction.tolist(),optimizer_success=bool(fit.success),
                scope='Bounded instantaneous linear acceleration problem; not full-trajectory or visual-occlusion feasibility')
