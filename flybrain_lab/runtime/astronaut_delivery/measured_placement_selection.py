"""Bounded destination proposal from measured marks, never free-space authority."""
import numpy as np


def spread_proposal(places, target_position, used_places=()):
    """Prefer dispersed table usage; near-tied scores prefer a shorter transfer.

    For the first object maximize summed distance to the observed alternatives
    (avoids occupying the middle of a row first). Afterwards maximize distance
    to the nearest previously measured release destination. Two-sigma center
    uncertainty defines a heuristic tie band, NOT a safety confidence bound.
    No object count, CAD position, fresh timestamp or collision permit is made.
    """
    places=list(places)
    if not places:raise RuntimeError('no_measured_placement_proposal')
    xy=np.asarray([e.position[:2] for e in places],float)
    target=np.asarray(target_position,float)
    used=np.asarray(used_places,float).reshape(-1,3)
    if xy.shape!=(len(places),2) or target.shape!=(3,) or not np.isfinite(xy).all() or not np.isfinite(target).all() or not np.isfinite(used).all():
        raise ValueError('finite measured destination coordinates required')
    sigma=[]
    for e in places:
        covariance=np.asarray(e.covariance,float)
        if covariance.shape!=(3,3) or not np.isfinite(covariance).all():raise ValueError('finite measured covariance required')
        eig=np.linalg.eigvalsh((covariance[:2,:2]+covariance[:2,:2].T)*.5)
        if eig[0]<-1e-12:raise ValueError('positive semidefinite measured covariance required')
        sigma.append(float(np.sqrt(max(0.,eig[-1]))))
    if len(used):
        scores=np.linalg.norm(xy[:,None,:]-used[None,:,:2],axis=2).min(axis=1)
        criterion='maximum_nearest_measured_used_destination_distance'
        terms=1
    else:
        scores=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2).sum(axis=1)
        criterion='maximum_sum_observed_mark_distances'
        terms=max(1,len(places)-1)
    tolerance=4*max(sigma)*terms
    eligible=np.flatnonzero(scores>=scores.max()-tolerance)
    selected=min(eligible,key=lambda i:(float(np.linalg.norm(xy[i]-target[:2])),str(places[i].name)))
    info=dict(criterion=criterion,score_tie_band_m=tolerance,
        selected=places[selected].name,used_destinations=len(used),
        candidates=[dict(name=e.name,position=e.position.tolist(),stamp_s=e.stamp,score_m=float(scores[i])) for i,e in enumerate(places)],
        scope='ROI proposal only; heuristic uncertainty tie band is not collision clearance or fresh visual evidence')
    return places[selected],info
