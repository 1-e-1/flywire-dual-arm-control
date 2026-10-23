"""16-variable acceleration QP with timestamped perception and a paired guard.

Normal actions obey q/v/a/jerk bounds. Safe braking may relax jerk to prioritize
collision avoidance. No actual robot transport or torque controller is present.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
import osqp
from .reaction import DifferentialArm,skew

@dataclass(frozen=True)
class VisualSnapshot:
    position:np.ndarray
    covariance:np.ndarray
    exposure_s:float
    available_s:float
    def __post_init__(self):
        for name in ('position','covariance'):
            value=np.array(getattr(self,name),float,copy=True);value.setflags(write=False);object.__setattr__(self,name,value)

class LatestObservation:
    """Latest-ready immutable message. Never turn exposure time into arrival time."""
    def __init__(self,max_age=.25):self.max_age=max_age;self.pending=[];self.current=None
    def publish(self,snapshot):
        p=np.asarray(snapshot.position,float);c=np.asarray(snapshot.covariance,float)
        if p.shape!=(3,) or c.shape!=(3,3) or not np.isfinite(np.r_[p,c.ravel(),snapshot.exposure_s,snapshot.available_s]).all():raise ValueError('Invalid visual snapshot')
        if not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min()<-1e-12:raise ValueError('Invalid visual covariance')
        if snapshot.available_s<snapshot.exposure_s:raise ValueError('Observation available before exposure')
        self.pending.append(snapshot)
    def read(self,now):
        ready=[s for s in self.pending if s.available_s<=now]
        self.pending=[s for s in self.pending if s.available_s>now]
        if ready:
            latest=max(ready,key=lambda s:s.exposure_s)
            if self.current is None or latest.exposure_s>self.current.exposure_s:self.current=latest
        s=self.current
        return s if s is not None and np.isfinite(now) and 0<=now-s.exposure_s<=self.max_age else None

class JointOnlineController:
    def __init__(self,left,right,reaction,paired_guard,cfg):
        self.arms=[left,right];self.trees=[DifferentialArm(a) for a in self.arms]
        self.reaction=reaction;self.guard=paired_guard;self.cfg=cfg
        self.previous_acceleration=np.zeros(16);self.timing=[];self.last_status='reset';self.solver=None
        self.rows=np.concatenate([np.arange(j+1) for j in range(16)]);self.cols=np.concatenate([np.full(j+1,j) for j in range(16)])

    def tasks(self,q,target):
        q=np.asarray(q).reshape(2,8);states=[tree.evaluate(x) for tree,x in zip(self.trees,q)]
        poses=[state.poses['b8']@arm.tool_local for state,arm in zip(states,self.arms)]
        right_p,Jv,Jw=states[1].point('b8',self.arms[1].tool_local[:3,3]);axis=poses[1][:3,2]
        Jr=np.zeros((6,16));Jr[:3,8:]=Jv;Jr[3:,8:]=-skew(axis)@Jw
        cp,Cv,Cw=states[0].point('b8',self.arms[0].tool_local[:3,3]);R=poses[0][:3,:3]@np.diag([1.,-1.,-1.])
        p=R.T@(target-cp);z=-p[2];distance=np.linalg.norm(target-cp)
        if z<=.05 or distance<=.05:raise ValueError('Target behind or too close to camera')
        dp=-R.T@Cv+R.T@skew(target-cp)@Cw
        project=np.array([[1/z,0,p[0]/(z*z)],[0,-1/z,-p[1]/(z*z)]])
        Jc=np.zeros((4,16));Jc[:2,:8]=project@dp;Jc[2,:8]=(cp-target)/distance@Cv
        # Existing task uses horizontal docking caps. Keep their normal visible;
        # centring alone can produce a grazing view with almost no useful depth.
        elevation=(cp[2]-target[2])/distance
        Jc[3,:8]=(np.array([0.,0.,1.])/distance-(cp[2]-target[2])*(cp-target)/(distance**3))@Cv
        return np.r_[right_p,axis],Jr,np.array([p[0]/z,-p[1]/z,distance,elevation]),Jc

    def propose(self,q,v,target,dt=None):
        start=time.perf_counter();cfg=self.cfg;dt=cfg['dt_s'] if dt is None else dt
        q=np.asarray(q).reshape(16);v=np.asarray(v).reshape(16);target=np.asarray(target)
        xr,Jr,xc,Jc=self.tasks(q,target)
        h=1e-5/max(1.,np.linalg.norm(v))
        plus=self.tasks(q+h*v,target);minus=self.tasks(q-h*v,target)
        br=(plus[1]-minus[1])@v/(2*h);bc=(plus[3]-minus[3])@v/(2*h)
        desired_r=np.r_[target+[0,0,cfg['right_target_height_m']],0,0,-1]
        desired_c=np.array([0.,0.,cfg['camera_target_range_m'],cfg['camera_preferred_elevation_sine']])
        ar=cfg['position_kp']*(desired_r-xr)-cfg['position_kd']*(Jr@v)-br
        ac=cfg['position_kp']*(desired_c-xc)-cfg['position_kd']*(Jc@v)-bc
        A,bias=self.reaction.coefficients(q,v)
        wr=np.array([100.,100.,100.,10.,10.,10.]);wc=np.array([10.,10.,3.,10.])
        scale=np.r_[np.full(3,1/cfg['force_scale_N']),np.full(3,1/cfg['torque_scale_Nm'])]
        wb=np.sqrt(cfg['reaction_weight'])*scale
        M=np.vstack([wr[:,None]*Jr,wc[:,None]*Jc,wb[:,None]*A,.02*np.eye(16)])
        y=np.r_[wr*ar,wc*ac,-wb*bias,.02*self.previous_acceleration]
        H=2*(M.T@M+1e-7*np.eye(16));g=-2*M.T@y
        vmax=cfg['maximum_velocity_rad_s'];amax=cfg['maximum_acceleration_rad_s2'];jerk=cfg['maximum_jerk_rad_s3']
        lower=np.maximum.reduce([np.full(16,-amax),(-vmax-v)/dt,2*(-np.pi-q-v*dt)/(dt*dt),self.previous_acceleration-jerk*dt])
        upper=np.minimum.reduce([np.full(16,amax),(vmax-v)/dt,2*(np.pi-q-v*dt)/(dt*dt),self.previous_acceleration+jerk*dt])
        # Reserve velocity headroom to slew acceleration back to zero with the
        # normal jerk bound. A one-step velocity clip alone can become infeasible.
        upper=np.minimum(upper,np.sqrt((jerk*dt)**2+2*jerk*np.maximum(0,vmax-v))-jerk*dt)
        lower=np.maximum(lower,-np.sqrt((jerk*dt)**2+2*jerk*np.maximum(0,vmax+v))+jerk*dt)
        if np.any(lower>upper):self.last_status='infeasible_joint_bounds';return None
        P=sparse.csc_matrix((H[self.rows,self.cols],(self.rows,self.cols)),shape=(16,16))
        feature_low=np.array([-.60,-.33,cfg['camera_range_limits_m'][0],cfg['camera_minimum_elevation_sine']])
        feature_high=np.array([.60,.33,cfg['camera_range_limits_m'][1],.90])
        clow=2*(feature_low-xc-dt*(Jc@v))/(dt*dt)-bc
        chigh=2*(feature_high-xc-dt*(Jc@v))/(dt*dt)-bc
        dense=np.vstack([np.eye(16),Jc]);ri=np.tile(np.arange(20),16);ci=np.repeat(np.arange(16),20)
        constraint=sparse.csc_matrix((dense.ravel(order='F'),(ri,ci)),shape=(20,16))
        bounds_low=np.r_[lower,clow];bounds_high=np.r_[upper,chigh]
        if self.solver is None:
            self.solver=osqp.OSQP();self.solver.setup(P=P,q=g,A=constraint,l=bounds_low,u=bounds_high,
                                                     verbose=False,warm_start=True,polish=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=4000)
        else:self.solver.update(Px=P.data,Ax=constraint.data,q=g,l=bounds_low,u=bounds_high)
        result=self.solver.solve()
        self.timing.append(time.perf_counter()-start)
        if result.info.status_val!=1 or result.x is None or not np.isfinite(result.x).all():self.last_status='qp_not_solved';return None
        a=np.asarray(result.x)
        if np.any(a<lower-1e-6) or np.any(a>upper+1e-6):self.last_status='qp_bound_violation';return None
        a=np.clip(a,lower,upper)
        # Range/visual screen is an endpoint gate; paired collision guard also
        # certifies the full curved step and an executable stopping trajectory.
        next_q=q+v*dt+.5*a*dt*dt;next_v=v+a*dt
        try:_,_,next_feature,_=self.tasks(next_q,target)
        except ValueError:self.last_status='invalid_camera_view';return None
        lo,hi=cfg['camera_range_limits_m']
        if not lo<=next_feature[2]<=hi or next_feature[3]<cfg['camera_minimum_elevation_sine']-1e-5:self.last_status='camera_range_or_incidence';return None
        if not self.guard.acceleration(q,v,a,dt):self.last_status='paired_collision_hold';return None
        if self.guard.brake(next_q,next_v,cfg['emergency_deceleration_rad_s2']) is None:self.last_status='brake_path_unavailable';return None
        self.previous_acceleration=a.copy();self.last_status='accepted'
        return a
