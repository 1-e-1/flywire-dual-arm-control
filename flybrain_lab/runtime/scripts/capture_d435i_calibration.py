"""Read-only-to-hardware D435i calibration/raw sample export; never drives arms.

Requires optional pyrealsense2 from the official RealSense SDK. Export is NOT a
robot hand-eye calibration and deliberately cannot unlock hardware execution.
"""
import argparse,json,time
from pathlib import Path
import numpy as np

def intrinsics_dict(profile):
    i=profile.as_video_stream_profile().get_intrinsics()
    return dict(width=i.width,height=i.height,fx=i.fx,fy=i.fy,ppx=i.ppx,ppy=i.ppy,
                distortion_model=str(i.model),coefficients=list(i.coeffs),fps=profile.fps())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--serial')
    args=parser.parse_args()
    # Never replace an existing unit calibration accidentally.
    if args.output.exists():raise SystemExit('Output exists; choose a new calibration directory.')
    try:import pyrealsense2 as rs
    except ImportError:raise SystemExit('pyrealsense2 is not installed. Install the official RealSense SDK Python binding in a hardware-acquisition environment; no hardware was accessed.')
    ctx=rs.context();devices=list(ctx.query_devices())
    devices=[d for d in devices if 'D435I' in d.get_info(rs.camera_info.name).upper() and (args.serial is None or d.get_info(rs.camera_info.serial_number)==args.serial)]
    if len(devices)!=1:raise SystemExit('Require exactly one selected D435i; use --serial if multiple devices are connected.')
    device=devices[0];serial=device.get_info(rs.camera_info.serial_number)
    pipeline=rs.pipeline(ctx);config=rs.config();config.enable_device(serial)
    config.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
    config.enable_stream(rs.stream.color,1280,720,rs.format.rgb8,30)
    started=False
    try:
        profile=pipeline.start(config);started=True
        for _ in range(30):frames=pipeline.wait_for_frames(5000)
        depth=frames.get_depth_frame();color=frames.get_color_frame()
        if not depth or not color:raise RuntimeError('Missing synchronized frames')
        dp=depth.profile.as_video_stream_profile();cp=color.profile.as_video_stream_profile()
        ex=dp.get_extrinsics_to(cp);T=np.eye(4)
        # rs2_extrinsics.rotation is column-major; this maps depth -> color.
        T[:3,:3]=np.array(ex.rotation).reshape(3,3,order='F');T[:3,3]=ex.translation
        scale=profile.get_device().first_depth_sensor().get_depth_scale()
        data=dict(status='DEVICE_INTRINSICS_EXPORTED_HAND_EYE_NOT_CALIBRATED',serial=serial,
                  firmware=device.get_info(rs.camera_info.firmware_version),depth=intrinsics_dict(dp),color=intrinsics_dict(cp),
                  depth_scale_m_per_unit=scale,T_color_from_depth_optical=T.tolist(),
                  optical_axes='X right, Y down, Z forward; convert to USD OpenGL with diag(1,-1,-1)',
                  host_monotonic_s=time.monotonic(),depth_timestamp_ms=depth.get_timestamp(),color_timestamp_ms=color.get_timestamp(),
                  depth_timestamp_domain=str(depth.get_frame_timestamp_domain()),color_timestamp_domain=str(color.get_frame_timestamp_domain()),
                  safety='No arm motion. Hand-eye, exposure-time robot pose interpolation, distortion handling and USB latency checks still required.')
        args.output.mkdir(parents=True)
        (args.output/'calibration.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        np.savez_compressed(args.output/'raw_rgbd.npz',depth_units=np.asanyarray(depth.get_data()),rgb=np.asanyarray(color.get_data()))
        print(json.dumps(data,indent=2))
    finally:
        if started:pipeline.stop()

if __name__=='__main__':main()
