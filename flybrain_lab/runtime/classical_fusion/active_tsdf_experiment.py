"""Reproducible calibration and budget-matched active-view replay experiment.

Replay candidates are previously executed robot poses. Selection sees only
poses/joints, NEVER unobserved depth or object GT. This evaluates view choice
and mapping, not unexecuted transitions or real-robot path safety.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from codex_zbot.perception.pointcloud import quaternion_to_matrix
from codex_zbot.tasks.shape_geometry import signed_distance_to_primitives
from classical_fusion.planning_map import PlanningTSDF,PlanningMapConfig
from classical_fusion.planning_audit import audit_snapshot


def calibrate(data_dir: Path, episodes=(0, 1),fit_principal=False) -> dict:
    """Fit a square-pixel camera on a known z=0 calibration plane.

    Semantic id 1 is used ONLY to select calibration-plane pixels in two
    designated calibration scenes. Production reconstruction and view scoring
    use no segmentation/GT; hardware would supply a calibration-board mask.
    """
    rows, targets = [], []
    image_shape = None
    for episode in episodes:
        with np.load(data_dir / f"ep_{episode:04d}.npz") as source:
            frames = np.arange(10, len(source["depth"]), 15)
            depth = source["depth"][frames, ..., 0]
            pose = source["pose"][frames]
            rotation = quaternion_to_matrix(torch.from_numpy(pose[:, 3:])).numpy()
            h, w = depth.shape[1:]
            image_shape = [h, w]
            v, u = np.mgrid[:h, :w]
            mask = (source["semantic"][frames, ..., 0] == 1) & np.isfinite(depth) & (depth > .1) & (depth < 2)
            coefficient = depth * ((u - w / 2) * rotation[:, None, None, 2, 1]
                                   + (v - h / 2) * rotation[:, None, None, 2, 2])
            target = pose[:, None, None, 2] + depth * rotation[:, None, None, 2, 0]
            if fit_principal:
                design=np.stack([
                    depth*(u-w/2)*rotation[:,None,None,2,1],
                    depth*(v-h/2)*rotation[:,None,None,2,2],
                    np.broadcast_to(depth*rotation[:,None,None,2,1],depth.shape),
                    np.broadcast_to(depth*rotation[:,None,None,2,2],depth.shape)],-1)
                rows.append(design[mask].astype(np.float64))
            else:
                rows.append(coefficient[mask].astype(np.float64))
            targets.append(target[mask].astype(np.float64))
    a, b = np.concatenate(rows), np.concatenate(targets)
    if fit_principal:
        solution=np.linalg.lstsq(a,b,rcond=None)[0]
        for _ in range(6):
            error=a@solution-b
            scale=max(.002,1.4826*float(np.median(np.abs(error))))
            weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(error),1e-12))
            solution=np.linalg.lstsq(a*np.sqrt(weights[:,None]),b*np.sqrt(weights),rcond=None)[0]
        fx,fy=1/solution[:2]
        h,w=image_shape
        cx,cy=w/2-solution[2]*fx,h/2-solution[3]*fy
        residual=float(np.mean(np.abs(a@solution-b)))
        if min(fx,fy)<=0 or residual>.01 or abs(cx-w/2)>5 or abs(cy-h/2)>5:
            raise RuntimeError('full intrinsic calibration failed validity checks')
        return {'intrinsics':[[fx,0,cx],[0,fy,cy],[0,0,1]],'image_shape':image_shape,
                'calibration_episodes':list(episodes),'plane_mean_absolute_residual_m':residual,
                'calibration_plane':'known z=0, independent fx/fy/cx/cy robust fit',
                'warning':'dataset-specific; do not transfer to other camera profiles'}
    inverse_f = float(a @ b / (a @ a))
    for _ in range(4):
        error = a * inverse_f - b
        keep = np.abs(error) < max(.006, 3 * 1.4826 * float(np.median(np.abs(error))))
        inverse_f = float(a[keep] @ b[keep] / (a[keep] @ a[keep]))
    h, w = image_shape
    focal = 1 / inverse_f
    if focal <= 0 or float(np.mean(np.abs(a * inverse_f - b))) > .01:
        raise RuntimeError("calibration plane does not support an accurate pinhole fit")
    return {
        "intrinsics": [[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]],
        "image_shape": image_shape,
        "calibration_episodes": list(episodes),
        "calibration_plane": "known world z=0; fixed square pixels and centered principal point",
        "plane_mean_absolute_residual_m": float(np.mean(np.abs(a * inverse_f - b))),
        "recorded_intrinsics": np.load(data_dir / "intrinsics.npy").tolist(),
        "warning": "Dataset-specific calibration; never transfer blindly to another camera or recollection.",
    }


def metrics(points: np.ndarray, source) -> dict:
    centers = torch.from_numpy(source["object_centers"].copy())[None]
    sizes = torch.from_numpy(source["object_sizes"].copy())[None]
    quaternions = torch.from_numpy(source["object_quaternions_wxyz"].copy())[None]
    def distance_to_truth(query):
        if "object_shape_ids" in source:
            return signed_distance_to_primitives(query,centers[0],sizes[0],quaternions[0],
                                                  source["object_shape_ids"])
        if "schema_version" in source and int(source["schema_version"]) >= 8:
            raise ValueError("mixed-shape schema requires shape IDs; refusing box-only metrics")
        return signed_distance_to_primitives(query,centers[0],sizes[0],quaternions[0],
                                             np.zeros(centers.shape[1],dtype=np.int64))
    if "gt_object_points" in source:
        truth=source["gt_object_points"].copy()
    else:
        ground_truth = torch.from_numpy(source["gt_points"].copy())
        truth_distance=distance_to_truth(ground_truth)
        truth = ground_truth[truth_distance.abs() < 1e-4].numpy()
    if not len(points) or not len(truth):
        return {"points": len(points), "precision_5mm": 0., "precision_10mm": 0.,
                "recall_5mm": 0., "recall_10mm": 0., "f1_10mm": 0., "surface_mae_m": None,
                "surface_p95_m":None,"surface_p99_m":None,"coverage_p95_m":None}
    distance = distance_to_truth(torch.from_numpy(points)).abs().numpy()
    nearest = cKDTree(points).query(truth, workers=1)[0]
    precision, recall = float((distance <= .01).mean()), float((nearest <= .01).mean())
    return {"points": len(points), "precision_5mm": float((distance <= .005).mean()),
            "precision_10mm": precision, "recall_5mm": float((nearest <= .005).mean()),
            "recall_10mm": recall, "f1_10mm": 2 * precision * recall / max(precision+recall, 1e-9),
            "surface_mae_m": float(distance.mean()),"surface_p95_m":float(np.quantile(distance,.95)),
            "surface_p99_m":float(np.quantile(distance,.99)),"coverage_p95_m":float(np.quantile(nearest,.95))}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sequences_v7_partial_visibility_formal")
    calibration_options=parser.add_mutually_exclusive_group()
    calibration_options.add_argument("--calibration_json",help="independently validated intrinsics for this camera/dataset")
    calibration_options.add_argument("--use_recorded_intrinsics",action="store_true",
                                    help="explicit diagnostic using recorded K; not independent calibration")
    parser.add_argument("--output_dir", default="classical_fusion/outputs/planning_v3")
    parser.add_argument("--start_episode", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--view_budget", type=int, default=8)
    parser.add_argument("--frames_per_view", type=int, default=5)
    parser.add_argument("--planning",action="store_true",default=True,
                        help="compatibility flag: the retained route is always Planning v3")
    parser.add_argument("--refine_calibration",action="store_true",help="fit fx/fy AND principal point on calibration scenes 0/1")
    parser.add_argument("--depth_input_clip_m",type=float,default=0.,help="explicit range-loss ablation; 0 keeps recorded valid depth")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=["tsdf_order","tsdf_random","tsdf_active"],default=["tsdf_active"])
    args = parser.parse_args()
    if args.start_episode < 0 or args.view_budget < 1 or args.frames_per_view < 1:
        parser.error("episode must be nonnegative and budgets positive")
    if not (args.calibration_json or args.use_recorded_intrinsics) and args.start_episode<2:
        parser.error("plane-fitted evaluation must exclude calibration episodes 0/1")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.calibration_json:
        calibration=json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    elif args.use_recorded_intrinsics:
        calibration={"intrinsics":np.load(data_dir/"intrinsics.npy").tolist(),
                     "warning":"recorded K explicitly requested; not an independently fitted camera calibration"}
    else:
        manifest_path=data_dir/"collection_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("schema_version",0)>=8:
            raise ValueError("new lens/shape data requires its own --calibration_json, or explicit --use_recorded_intrinsics diagnostic")
        calibration = calibrate(data_dir,fit_principal=args.refine_calibration)
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    k = torch.tensor(calibration["intrinsics"], dtype=torch.float32, device=args.device)
    config=PlanningMapConfig()
    records = {mode: [] for mode in args.modes}
    print("calibration",calibration["intrinsics"],"plane MAE",calibration.get("plane_mean_absolute_residual_m"),flush=True)
    for ep in range(args.start_episode, args.start_episode + args.episodes):
        with np.load(data_dir / f"ep_{ep:04d}.npz", allow_pickle=False) as source:
            # Stable end of each executed view; reject transitions, not based
            # on future depth content. Cached candidates are joint/FK only.
            stops = np.r_[source["view_change_frames"] - 1, len(source["pose"])-1]
            poses = torch.from_numpy(source["pose"][stops].copy()).to(args.device)
            joints = torch.from_numpy(source["joint_pos"][stops].copy()).to(args.device)
            depth = source["depth"]
            robot_pose = source["pose"]
            for mode in args.modes:
                mapper = PlanningTSDF(config,args.device)
                matrix = k
                random = np.random.default_rng(args.seed + ep)
                random_order = [0] + random.permutation(np.arange(1, len(stops))).tolist()
                selected, updates, scoring = [], [], []
                path_l2 = 0.
                for step in range(min(args.view_budget, len(stops))):
                    if step == 0:
                        candidate = 0
                    elif mode == "tsdf_active":
                        reachable = torch.ones(len(stops), dtype=torch.bool, device=args.device)
                        reachable[selected] = False
                        synchronize(args.device); before = time.perf_counter()
                        scores = mapper.view_scores(poses, k, depth.shape[1:3], joints=joints,
                                                    current_joints=joints[selected[-1]], reachable=reachable)
                        synchronize(args.device); scoring.append(time.perf_counter()-before)
                        if not torch.isfinite(scores).any():
                            break
                        candidate = int(scores.argmax())
                    else:
                        candidate = random_order[step] if mode == "tsdf_random" else step
                    if selected:
                        path_l2 += float((joints[candidate]-joints[selected[-1]]).norm())
                    selected.append(candidate)
                    stop = int(stops[candidate])
                    for frame in range(stop-args.frames_per_view+1, stop+1):
                        d = torch.from_numpy(depth[frame].copy()).to(args.device)
                        if args.depth_input_clip_m>0:
                            d=torch.where(d<args.depth_input_clip_m,d,0.)
                        pose = torch.from_numpy(robot_pose[frame].copy()).to(args.device)
                        synchronize(args.device); before = time.perf_counter()
                        mapper.integrate(d, matrix, pose)
                        synchronize(args.device); updates.append(time.perf_counter()-before)
                synchronize(args.device); before = time.perf_counter()
                cloud, confidence = mapper.surface()
                synchronize(args.device); extract_seconds = time.perf_counter()-before
                points = cloud.cpu().numpy()
                result = metrics(points, source)
                result.update(episode=ep, selected_views=selected, joint_path_l2=path_l2,
                              mean_integration_ms=1000*float(np.mean(updates)),
                              surface_extraction_ms=1000*extract_seconds,
                              mean_scoring_ms=1000*float(np.mean(scoring)) if scoring else 0.)
                if args.planning:
                    result.update(audit_snapshot(mapper,source,args.seed+ep,
                                  artifact_dir=output_dir if ep==args.start_episode else None))
                records[mode].append(result)
                if ep == args.start_episode:
                    np.savez_compressed(output_dir / f"sample_{mode}.npz", points=points,
                                        confidence=confidence.cpu().numpy(), views=selected)
                    if mode == "tsdf_active":
                        synchronize(args.device); before = time.perf_counter()
                        vertices, faces = mapper.mesh()
                        synchronize(args.device)
                        mesh_ms = 1000*(time.perf_counter()-before)
                        gradient, hessian, derivative_valid = mapper.differentials(vertices)
                        np.savez_compressed(output_dir / "sample_mesh.npz",
                                            vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy(),
                                            gradient=gradient.cpu().numpy(), hessian=hessian.cpu().numpy(),
                                            derivative_valid=derivative_valid.cpu().numpy())
                        (output_dir / "mesh_timing.json").write_text(json.dumps(
                            {"milliseconds": mesh_ms, "vertices": len(vertices), "faces": len(faces),
                             "unknown_boundaries": "left open, not assumed free or filled"},indent=2))
                print(ep, mode, "P10", round(result["precision_10mm"],3), "R10",round(result["recall_10mm"],3),
                      "ms",round(result["mean_integration_ms"],2), "views",selected, flush=True)
    summary = {}
    for mode, runs in records.items():
        keys = [key for key in runs[0] if key not in ("episode", "selected_views")]
        summary[mode]={}
        for key in keys:
            values=[r[key] for r in runs if r[key] is not None]
            summary[mode][key]=float(np.mean(values)) if values else None
    report = {"protocol": "fixed view budget, held-out calibration scenes, FK-pose candidate replay; not a live robot trial",
              "config": config.to_dict(), "calibration": calibration, "arguments": vars(args),
              "summary": summary, "episodes": records,
              "ground_truth": "only calibration plane mask and post-hoc object surface metrics; no GT in mapping or view scores"}
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
