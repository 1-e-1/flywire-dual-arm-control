"""Action-local map invalidation, with replayable events for latest-frame IPC."""
import numpy as np


def action_region(center,half_extent=.14):
    # Task prior: every part is contained within 80 mm of the measured cap or
    # commanded object origin. 36 mm TSDF band + 24 mm pose/error allowance.
    # This bound is NOT inferred from hidden simulator object poses.
    c=np.asarray(center,dtype=float)
    if c.shape!=(3,) or not np.isfinite(c).all() or half_extent<.14:
        raise ValueError('invalid or insufficient action envelope')
    return dict(lower=(c-half_extent).tolist(),upper=(c+half_extent).tolist())


def mask_carried_depth(obs,tcp_world,radius=.11):
    """Exclude measured points in the known grasp envelope, including grey body.

    Uses encoders/FK + task size bound, never renderer instance IDs or part pose.
    Background in this small envelope is conservatively unobserved, not freed.
    """
    center=(np.asarray(tcp_world)-obs.camera_world[:3,3])@obs.camera_world[:3,:3]
    v,u=np.indices(obs.depth.shape);z=obs.depth
    squared=((u-obs.K[0,2])/obs.K[0,0]*z-center[0])**2
    squared+=(-(v-obs.K[1,2])/obs.K[1,1]*z-center[1])**2+(-z-center[2])**2
    mask=(z>0)&np.isfinite(z)&(squared<=radius*radius)
    obs.robot_mask=mask if obs.robot_mask is None else obs.robot_mask|mask
    return int(mask.sum())


def replay_changes(guide,current_epoch,new_epoch,changes):
    """Apply ALL missed action events; frame dropping must not drop map edits."""
    if new_epoch<current_epoch:raise ValueError('map epoch cannot move backwards')
    if new_epoch==current_epoch:return current_epoch,[]
    pending=[e for e in changes if current_epoch<e['epoch']<=new_epoch]
    if [e['epoch'] for e in pending]!=list(range(current_epoch+1,new_epoch+1)):
        guide.invalidate_after_transfer()
        return new_epoch,[dict(policy='reset',reason='missing_action_history',epoch=new_epoch)]
    applied=[]
    for event in pending:
        stats=guide.invalidate_after_transfer(event.get('regions'),event.get('target'))
        applied.append(dict(**event,statistics=stats))
    return new_epoch,applied
