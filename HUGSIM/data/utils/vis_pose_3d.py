import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class FramePose:
    rgb_path: str
    camera_name: str
    timestamp: float
    c2w: np.ndarray
    intr: np.ndarray
    width: int
    height: int


def get_opts():
    parser = argparse.ArgumentParser("Visualize prior/COLMAP poses in Open3D")
    parser.add_argument(
        "--datapath",
        type=str,
        default="",
        help="Single scene output dir (legacy mode)",
    )
    parser.add_argument(
        "--prior_datapath",
        type=str,
        default="",
        help="Prior scene dir (compare mode). If set, uses this for prior poses.",
    )
    parser.add_argument(
        "--optimized_datapath",
        type=str,
        default="",
        help="Optimized scene dir (compare mode). If set, uses this for optimized poses.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="prior",
        choices=["prior", "optimized", "both"],
        help="Pose source to visualize",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="CAM_FRONT",
        help="Camera name (e.g., CAM_FRONT) or all",
    )
    parser.add_argument("--t0", type=float, default=0.0, help="Start timestamp (sec)")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration from t0 in sec (<=0 means until end)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride")
    parser.add_argument(
        "--frustum_scale",
        type=float,
        default=1.5,
        help="Frustum depth scale in meters",
    )
    parser.add_argument(
        "--forward_axis",
        type=str,
        default="z",
        choices=["x", "z"],
        help="Axis used for heading smoothness checks",
    )
    parser.add_argument(
        "--point_cloud",
        type=str,
        default="",
        help="Optional point cloud path (.ply). If empty, auto-detect in datapath.",
    )
    parser.add_argument(
        "--prior_point_cloud",
        type=str,
        default="",
        help="Optional prior point cloud path in compare mode.",
    )
    parser.add_argument(
        "--optimized_point_cloud",
        type=str,
        default="",
        help="Optional optimized point cloud path in compare mode.",
    )
    parser.add_argument("--max_points", type=int, default=300000, help="Point cloud cap")
    parser.add_argument("--no_pointcloud", action="store_true", help="Hide point cloud")
    parser.add_argument("--no_stats", action="store_true", help="Skip numeric diagnostics")
    parser.add_argument(
        "--traj_only",
        action="store_true",
        help="Show only camera-center trajectory lines (hide frustums).",
    )
    parser.add_argument(
        "--frustum_stride",
        type=int,
        default=1,
        help="Draw one frustum every N frames.",
    )
    parser.add_argument(
        "--max_traj_step",
        type=float,
        default=0.0,
        help="If >0, do not connect trajectory edges longer than this distance (m).",
    )
    parser.add_argument(
        "--gps_csv",
        type=str,
        default="",
        help="Optional GPS CSV path (expects lat_deg, lon_deg, alt_m columns).",
    )
    parser.add_argument(
        "--gps_hz",
        type=float,
        default=5576.0 / 233.0,
        help="GPS sampling rate in Hz.",
    )
    parser.add_argument(
        "--gps_start_sec",
        type=float,
        default=0.0,
        help="Start time in GPS log that maps to reference frame timestamp 0.",
    )
    parser.add_argument(
        "--gps_ref_lat",
        type=float,
        default=37.601,
        help="Reference latitude for equirectangular lon/lat->meter projection.",
    )
    parser.add_argument(
        "--gps_use_alt",
        action="store_true",
        help="Use CSV altitude for Z axis. Default is flat z=0 trajectory.",
    )
    parser.add_argument(
        "--gps_align_to",
        type=str,
        default="prior",
        choices=["none", "prior", "optimized"],
        help="Rigidly align GPS XY trajectory to selected source for easier comparison.",
    )
    parser.add_argument(
        "--gps_source_name",
        type=str,
        default="gps",
        help="Label used for GPS source in logs/colors.",
    )
    parser.add_argument(
        "--gps_yaw_min_step",
        type=float,
        default=0.2,
        help="Minimum XY displacement (m) used to estimate stable GPS heading.",
    )
    parser.add_argument(
        "--gps_yaw_smooth",
        type=int,
        default=7,
        help="Moving-average window for GPS yaw smoothing (odd number recommended).",
    )
    parser.add_argument(
        "--gps_pos_smooth",
        type=int,
        default=5,
        help="Moving-average window for GPS XY smoothing before heading estimation.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Skip launching Open3D window (useful on headless/Wayland servers).",
    )
    return parser.parse_args()


def infer_camera_name(rgb_path: str) -> str:
    # Expected shape: ./images/CAM_FRONT/00000.jpg
    parts = Path(rgb_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def load_frames(meta_path: str):
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    frames = []
    for frame in meta_data["frames"]:
        frames.append(
            FramePose(
                rgb_path=frame["rgb_path"],
                camera_name=infer_camera_name(frame["rgb_path"]),
                timestamp=float(frame.get("timestamp", 0.0)),
                c2w=np.asarray(frame["camtoworld"], dtype=np.float64),
                intr=np.asarray(frame["intrinsics"], dtype=np.float64),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
        )
    return frames


def load_gps_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        req = ["lat_deg", "lon_deg", "alt_m"]
        for key in req:
            if key not in reader.fieldnames:
                raise ValueError(f"GPS CSV missing required column: {key}")
        for row in reader:
            rows.append(
                {
                    "lat": float(row["lat_deg"]),
                    "lon": float(row["lon_deg"]),
                    "alt": float(row["alt_m"]),
                }
            )
    if len(rows) == 0:
        raise ValueError(f"GPS CSV has no rows: {csv_path}")
    return rows


def latlon_to_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, ref_lat_deg: float):
    # Same projection family used in make_prior.py (equirectangular approximation).
    earth_r = 6378137.0
    x = earth_r * np.radians(lon_deg) * np.cos(np.radians(ref_lat_deg))
    y = earth_r * np.radians(lat_deg)
    return x, y


def moving_average_1d(x: np.ndarray, win: int):
    win = max(1, int(win))
    if win <= 1 or len(x) <= 2:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x_pad, kernel, mode="valid")


def compute_yaw_from_positions(positions_xyz: np.ndarray, min_step_m: float = 0.2):
    n = len(positions_xyz)
    yaws = np.zeros(n, dtype=np.float64)
    if n == 1:
        return yaws
    min_step_m = max(0.0, float(min_step_m))
    prev = 0.0
    for i in range(n):
        jf = i + 1
        while jf < n:
            d = positions_xyz[jf, :2] - positions_xyz[i, :2]
            if np.linalg.norm(d) >= min_step_m:
                break
            jf += 1

        jb = i - 1
        while jb >= 0:
            d = positions_xyz[i, :2] - positions_xyz[jb, :2]
            if np.linalg.norm(d) >= min_step_m:
                break
            jb -= 1

        if jb >= 0 and jf < n:
            dxy = positions_xyz[jf, :2] - positions_xyz[jb, :2]
        elif jf < n:
            dxy = positions_xyz[jf, :2] - positions_xyz[i, :2]
        elif jb >= 0:
            dxy = positions_xyz[i, :2] - positions_xyz[jb, :2]
        else:
            dxy = np.array([0.0, 0.0], dtype=np.float64)

        if np.linalg.norm(dxy) < 1e-9:
            yaws[i] = prev
        else:
            yaws[i] = np.arctan2(dxy[1], dxy[0])
            prev = yaws[i]
    return yaws


def make_c2w_from_yaw_translation(yaw: float, t_xyz: np.ndarray):
    c = np.cos(yaw)
    s = np.sin(yaw)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    c2w[:3, 3] = t_xyz
    return c2w


def fit_rigid_2d(src_xy: np.ndarray, dst_xy: np.ndarray):
    if len(src_xy) != len(dst_xy):
        raise ValueError("fit_rigid_2d requires equal number of points")
    if len(src_xy) < 2:
        return np.eye(2), np.zeros(2)
    mu_s = src_xy.mean(axis=0)
    mu_d = dst_xy.mean(axis=0)
    s0 = src_xy - mu_s
    d0 = dst_xy - mu_d
    h = s0.T @ d0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    return r, t


def apply_rigid_2d_to_frames(frames, rot2d: np.ndarray, trans2d: np.ndarray):
    out = []
    rot3 = np.eye(3, dtype=np.float64)
    rot3[:2, :2] = rot2d
    for frame in frames:
        c2w = frame.c2w.copy()
        c2w[:3, :3] = rot3 @ c2w[:3, :3]
        xy = c2w[:2, 3]
        c2w[0, 3], c2w[1, 3] = (rot2d @ xy + trans2d)
        out.append(
            FramePose(
                rgb_path=frame.rgb_path,
                camera_name=frame.camera_name,
                timestamp=frame.timestamp,
                c2w=c2w,
                intr=frame.intr,
                width=frame.width,
                height=frame.height,
            )
        )
    return out


def unique_timestamps_with_template(frames):
    seen = set()
    out = []
    for frame in sorted(frames, key=lambda x: (x.timestamp, x.rgb_path)):
        key = round(float(frame.timestamp), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(frame)
    return out


def build_gps_frames(
    csv_path,
    gps_hz,
    gps_start_sec,
    ref_frames,
    ref_lat,
    use_alt,
    yaw_min_step,
    yaw_smooth,
    pos_smooth,
):
    rows = load_gps_rows(csv_path)
    lat = np.array([r["lat"] for r in rows], dtype=np.float64)
    lon = np.array([r["lon"] for r in rows], dtype=np.float64)
    alt = np.array([r["alt"] for r in rows], dtype=np.float64)
    x, y = latlon_to_xy_m(lat, lon, ref_lat)
    z = alt if use_alt else np.zeros_like(alt)

    gps_t = np.arange(len(rows), dtype=np.float64) / float(gps_hz)
    ref_unique = unique_timestamps_with_template(ref_frames)
    ref_times = np.array([f.timestamp for f in ref_unique], dtype=np.float64)
    if len(ref_times) == 0:
        raise ValueError("No reference frames for GPS interpolation")
    query_t = gps_start_sec + (ref_times - ref_times[0])

    min_t = float(gps_t[0])
    max_t = float(gps_t[-1])
    if query_t.min() < min_t or query_t.max() > max_t:
        raise ValueError(
            f"GPS query time out of range. query=[{query_t.min():.3f}, {query_t.max():.3f}] "
            f"gps=[{min_t:.3f}, {max_t:.3f}]. Adjust --gps_start_sec or duration."
        )

    ix = np.interp(query_t, gps_t, x)
    iy = np.interp(query_t, gps_t, y)
    iz = np.interp(query_t, gps_t, z)
    ix = moving_average_1d(ix, pos_smooth)
    iy = moving_average_1d(iy, pos_smooth)
    pos = np.stack([ix, iy, iz], axis=1)
    pos -= pos[0]
    yaws = compute_yaw_from_positions(pos, min_step_m=yaw_min_step)
    yaws = moving_average_1d(np.unwrap(yaws), yaw_smooth)

    gps_frames = []
    for i, ref in enumerate(ref_unique):
        c2w = make_c2w_from_yaw_translation(float(yaws[i]), pos[i])
        gps_frames.append(
            FramePose(
                rgb_path=f"./gps/{i:06d}.csv",
                camera_name=ref.camera_name,
                timestamp=float(ref.timestamp),
                c2w=c2w,
                intr=ref.intr.copy(),
                width=ref.width,
                height=ref.height,
            )
        )
    return gps_frames


def filter_frames(frames, camera_name: str, t0: float, duration: float, stride: int):
    t1 = np.inf if duration <= 0 else t0 + duration
    filtered = []
    for frame in frames:
        if camera_name != "all" and frame.camera_name != camera_name:
            continue
        if frame.timestamp < t0 or frame.timestamp > t1:
            continue
        filtered.append(frame)

    filtered.sort(key=lambda x: (x.timestamp, x.rgb_path))
    stride = max(1, int(stride))
    return filtered[::stride]


def make_trajectory_lineset(frames, color, max_step=0.0):
    if len(frames) < 2:
        return None
    centers = np.array([f.c2w[:3, 3] for f in frames], dtype=np.float64)
    lines = []
    for i in range(len(centers) - 1):
        if max_step > 0:
            step = np.linalg.norm(centers[i + 1] - centers[i])
            if step > max_step:
                continue
        lines.append([i, i + 1])
    if len(lines) == 0:
        return None
    line_colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (len(lines), 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centers)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def make_camera_frustum(frame: FramePose, color, scale=1.5):
    fx = frame.intr[0, 0]
    fy = frame.intr[1, 1]
    cx = frame.intr[0, 2]
    cy = frame.intr[1, 2]
    w = float(frame.width)
    h = float(frame.height)

    corners = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0],
        ],
        dtype=np.float64,
    )
    z = float(scale)
    x = (corners[:, 0] - cx) * z / fx
    y = (corners[:, 1] - cy) * z / fy
    cam_pts = np.stack([x, y, np.full_like(x, z)], axis=1)
    cam_pts = np.concatenate([np.zeros((1, 3)), cam_pts], axis=0)

    world_pts = (frame.c2w[:3, :3] @ cam_pts.T).T + frame.c2w[:3, 3]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]
    line_colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(world_pts)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(line_colors)
    return frustum


def choose_point_cloud(datapath: str, point_cloud_arg: str):
    if point_cloud_arg:
        return point_cloud_arg
    candidates = [
        "ground_lidar.ply",
        "sparse_ba.ply",
        "points3d.ply",
        "ground_points3d.ply",
    ]
    for rel in candidates:
        p = os.path.join(datapath, rel)
        if os.path.exists(p):
            return p
    return ""


def load_point_cloud(point_path: str, max_points: int):
    if not point_path or not os.path.exists(point_path):
        return None
    pcd = o3d.io.read_point_cloud(point_path)
    if pcd.is_empty():
        return None
    n_points = len(pcd.points)
    if max_points > 0 and n_points > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_points, size=max_points, replace=False)
        pcd = pcd.select_by_index(idx.tolist())
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    return pcd


def angle_between_rotations_deg(r1: np.ndarray, r2: np.ndarray):
    rel = r1.T @ r2
    cos_theta = (np.trace(rel) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def print_pose_stats(frames, name: str, forward_axis: str):
    if len(frames) < 2:
        print(f"[{name}] not enough frames for diagnostics")
        return

    axis_idx = 2 if forward_axis == "z" else 0
    grouped = {}
    for frame in frames:
        grouped.setdefault(frame.camera_name, []).append(frame)

    print(f"[{name}] Diagnostics (grouped by camera)")
    for cam, seq in sorted(grouped.items()):
        if len(seq) < 2:
            continue
        seq.sort(key=lambda x: (x.timestamp, x.rgb_path))
        centers = np.array([f.c2w[:3, 3] for f in seq], dtype=np.float64)
        rots = [f.c2w[:3, :3] for f in seq]
        ts = np.array([f.timestamp for f in seq], dtype=np.float64)

        delta_t = np.diff(ts)
        delta_trans = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        delta_rot = np.array(
            [angle_between_rotations_deg(rots[i], rots[i + 1]) for i in range(len(rots) - 1)],
            dtype=np.float64,
        )

        fwd = np.array([r[:, axis_idx] for r in rots], dtype=np.float64)
        yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
        delta_yaw = np.degrees(np.abs(np.diff(yaw)))

        print(
            f"  - {cam}: n={len(seq)}, dt_mean={delta_t.mean():.4f}s, "
            f"trans_mean={delta_trans.mean():.4f}m, trans_max={delta_trans.max():.4f}m, "
            f"rot_max={delta_rot.max():.2f}deg, yaw_max={delta_yaw.max():.2f}deg"
        )

        # Show top-3 spikes to quickly inspect problematic frames.
        top_k = min(3, len(delta_trans))
        if top_k > 0:
            top_trans_idx = np.argsort(delta_trans)[-top_k:][::-1]
            top_rot_idx = np.argsort(delta_rot)[-top_k:][::-1]
            for idx in top_trans_idx:
                print(
                    f"      trans spike @ t={ts[idx+1]:.3f}s "
                    f"(from {seq[idx].rgb_path} -> {seq[idx+1].rgb_path}): {delta_trans[idx]:.4f}m"
                )
            for idx in top_rot_idx:
                print(
                    f"      rot spike @ t={ts[idx+1]:.3f}s "
                    f"(from {seq[idx].rgb_path} -> {seq[idx+1].rgb_path}): {delta_rot[idx]:.2f}deg"
                )


def resolve_sources(args):
    source_to_meta = {}
    source_to_pcd = {}

    # Compare mode: prior/optimized come from different scene dirs.
    if args.prior_datapath or args.optimized_datapath:
        if args.prior_datapath:
            prior_path = os.path.join(args.prior_datapath, "meta_data_init.json")
            if not os.path.exists(prior_path):
                prior_path = os.path.join(args.prior_datapath, "meta_data.json")
            source_to_meta["prior"] = prior_path
            source_to_pcd["prior"] = choose_point_cloud(
                args.prior_datapath, args.prior_point_cloud
            )
        if args.optimized_datapath:
            source_to_meta["optimized"] = os.path.join(
                args.optimized_datapath, "meta_data.json"
            )
            source_to_pcd["optimized"] = choose_point_cloud(
                args.optimized_datapath, args.optimized_point_cloud
            )
        return source_to_meta, source_to_pcd

    if not args.datapath:
        raise ValueError(
            "Please set --datapath (single mode) or --prior_datapath/--optimized_datapath (compare mode)."
        )

    if args.source in ("prior", "both"):
        prior_path = os.path.join(args.datapath, "meta_data_init.json")
        if not os.path.exists(prior_path):
            prior_path = os.path.join(args.datapath, "meta_data.json")
        source_to_meta["prior"] = prior_path
    if args.source in ("optimized", "both"):
        source_to_meta["optimized"] = os.path.join(args.datapath, "meta_data.json")
    if args.source in ("prior", "both"):
        source_to_pcd["prior"] = choose_point_cloud(args.datapath, args.point_cloud)
    if args.source in ("optimized", "both"):
        source_to_pcd["optimized"] = choose_point_cloud(args.datapath, args.point_cloud)

    return source_to_meta, source_to_pcd


def main():
    args = get_opts()
    if o3d is None:
        raise ImportError(
            "open3d is not installed in this environment. "
            "Install it first (e.g., pip install open3d) and rerun."
        )

    colors = {
        "prior": np.array([0.95, 0.25, 0.25]),      # red
        "optimized": np.array([0.10, 0.80, 0.25]),  # green
    }

    source_to_meta, source_to_pcd = resolve_sources(args)
    all_geometries = []
    frames_by_source = {}

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    all_geometries.append(world_axis)

    for source_name, meta_path in source_to_meta.items():
        if not os.path.exists(meta_path):
            print(f"[{source_name}] missing meta file: {meta_path}")
            continue

        frames = load_frames(meta_path)
        frames = filter_frames(frames, args.camera, args.t0, args.duration, args.stride)
        print(f"[{source_name}] meta={meta_path}")
        print(f"[{source_name}] selected frames: {len(frames)}")
        if len(frames) == 0:
            continue
        frames_by_source[source_name] = frames

    # Optional GPS source from CSV (prior in raw sensor space).
    if args.gps_csv:
        if len(frames_by_source) == 0:
            raise ValueError("GPS visualization requires at least one non-GPS source as reference.")

        # Use selected alignment source as template for time/intrinsics when available.
        if args.gps_align_to in frames_by_source:
            template_source = args.gps_align_to
        else:
            template_source = next(iter(frames_by_source.keys()))

        gps_frames = build_gps_frames(
            csv_path=args.gps_csv,
            gps_hz=args.gps_hz,
            gps_start_sec=args.gps_start_sec,
            ref_frames=frames_by_source[template_source],
            ref_lat=args.gps_ref_lat,
            use_alt=args.gps_use_alt,
            yaw_min_step=args.gps_yaw_min_step,
            yaw_smooth=args.gps_yaw_smooth,
            pos_smooth=args.gps_pos_smooth,
        )

        if args.gps_align_to != "none" and args.gps_align_to in frames_by_source:
            ref = unique_timestamps_with_template(frames_by_source[args.gps_align_to])
            gps_u = unique_timestamps_with_template(gps_frames)
            n = min(len(ref), len(gps_u))
            src_xy = np.array([f.c2w[:2, 3] for f in gps_u[:n]], dtype=np.float64)
            dst_xy = np.array([f.c2w[:2, 3] for f in ref[:n]], dtype=np.float64)
            rot2d, trans2d = fit_rigid_2d(src_xy, dst_xy)
            gps_frames = apply_rigid_2d_to_frames(gps_frames, rot2d, trans2d)
            print(f"[gps] aligned to {args.gps_align_to} using 2D rigid fit (n={n})")

        gps_name = args.gps_source_name.strip() or "gps"
        frames_by_source[gps_name] = gps_frames
        colors[gps_name] = np.array([0.15, 0.45, 0.95])  # blue
        print(f"[{gps_name}] csv={args.gps_csv}")
        print(f"[{gps_name}] selected frames: {len(gps_frames)}")

    for source_name, frames in frames_by_source.items():
        if not args.no_pointcloud and source_name in source_to_pcd:
            pcd_path = source_to_pcd.get(source_name, "")
            pcd = load_point_cloud(pcd_path, args.max_points)
            if pcd is not None:
                # Tint each source cloud to make overlap easier to inspect.
                pcd.paint_uniform_color([0.6, 0.6, 0.6])
                all_geometries.append(pcd)
                print(f"[{source_name}] loaded point cloud: {pcd_path} (points={len(pcd.points)})")
            else:
                print(f"[{source_name}] no valid point cloud loaded.")

        if not args.no_stats:
            print_pose_stats(frames, source_name, args.forward_axis)

        traj = make_trajectory_lineset(
            frames,
            color=colors[source_name],
            max_step=args.max_traj_step,
        )
        if traj is not None:
            all_geometries.append(traj)

        if not args.traj_only:
            fstride = max(1, int(args.frustum_stride))
            for frame in frames[::fstride]:
                frustum = make_camera_frustum(
                    frame,
                    color=colors[source_name],
                    scale=args.frustum_scale,
                )
                all_geometries.append(frustum)

    if len(all_geometries) <= 1:
        print("Nothing to visualize. Check filters (--camera/--t0/--duration) and input files.")
        return

    if args.no_viewer:
        print("Skipping Open3D viewer (--no_viewer).")
        return

    print("Launching Open3D viewer...")
    o3d.visualization.draw_geometries(all_geometries)


if __name__ == "__main__":
    main()
