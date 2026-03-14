import os
import glob
import re
import numpy as np
import cv2
import OpenEXR, Imath
import flow_vis
import sys

# -----------------------
# EXR reading utilities
# -----------------------

def _exr_size(exr):
    hdr = exr.header()
    dw = hdr["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    return H, W, hdr

def read_exr_channels(path, prefer_groups):
    """
    Read one or more channels from an EXR.

    prefer_groups: list of tuples/lists, each a channel name group to try in order.
      Example for depth:   [("R",), ("Z",), ("Depth",)]
      Example for normals: [("R","G","B"), ("X","Y","Z")]
      Example for flow:    [("R","A"), ("R","G"), ("X","Y")]

    Returns:
      np.ndarray shape (H, W) for 1 ch, or (H, W, C) for multi-ch, dtype float32
    """
    exr = OpenEXR.InputFile(path)
    H, W, hdr = _exr_size(exr)
    chans = set(hdr["channels"].keys())

    chosen = None
    for group in prefer_groups:
        if all(c in chans for c in group):
            chosen = group
            break

    if chosen is None:
        allc = sorted(list(chans))
        need = len(prefer_groups[0])
        chosen = tuple(allc[:need])

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    arrays = []
    for c in chosen:
        raw = exr.channel(c, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        arrays.append(arr)

    if len(arrays) == 1:
        return arrays[0]
    return np.stack(arrays, axis=-1)

# -----------------------
# Visualization helpers
# -----------------------

def depth_to_u8(depth, p_lo=1, p_hi=99):
    depth = depth.copy()
    depth[~np.isfinite(depth)] = np.nan

    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)

    lo, hi = np.nanpercentile(depth, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)

    d01 = (depth - lo) / (hi - lo)
    d01 = np.clip(d01, 0, 1)
    return (255 * d01).astype(np.uint8)

def normals_to_bgr_u8(normals):
    n = normals.copy()
    n[~np.isfinite(n)] = 0.0
    nmin, nmax = np.nanmin(n), np.nanmax(n)

    if nmin >= 0.0 and nmax <= 1.0:
        mapped = np.clip(n, 0, 1)
    else:
        mapped = np.clip(0.5 * (n + 1.0), 0, 1)

    u8 = (255 * mapped).astype(np.uint8)
    return u8[..., ::-1]  # RGB -> BGR

def flow_to_bgr_u8(flow_uv, mag_p=99):
    f = flow_uv.copy()
    f[~np.isfinite(f)] = 0.0
    u = f[..., 0]
    v = f[..., 1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)

    hue = ((ang + np.pi) * (179.0 / (2.0 * np.pi))).astype(np.uint8)

    m = mag.copy()
    m[~np.isfinite(m)] = 0.0
    scale = np.percentile(m, mag_p) if np.any(m > 0) else 1.0
    if scale <= 1e-8:
        scale = 1.0

    val_u8 = (255 * np.clip(m / scale, 0, 1)).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)

    hsv = cv2.merge([hue, sat, val_u8])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# -----------------------
# New output patterns
# -----------------------

BASE_DIR = sys.argv[1]
OUT_MP4  = os.path.join(BASE_DIR, sys.argv[2]+".mp4")
FPS = 30

rgb_files   = glob.glob(os.path.join(BASE_DIR, "*_RGB.png"))
depth_files = glob.glob(os.path.join(BASE_DIR, "*_Depth.exr"))
norm_files  = glob.glob(os.path.join(BASE_DIR, "*_Normal.exr"))
flow_files  = glob.glob(os.path.join(BASE_DIR, "*_Optic_Flow.exr"))

assert rgb_files,   "No *_RGB.png frames found."
assert depth_files, "No *_Depth.exr frames found."
assert norm_files,  "No *_Normal.exr frames found."
assert flow_files,  "No *_Optic_Flow.exr frames found."

# Build frame-indexed maps so files can’t get out of sync
frame_re = re.compile(r"(\d{4})_")  # matches '0001_' at start of filename

def index_by_frame(files):
    d = {}
    for f in files:
        m = frame_re.search(os.path.basename(f))
        if not m:
            continue
        d[int(m.group(1))] = f
    return d

rgb_map   = index_by_frame(rgb_files)
depth_map = index_by_frame(depth_files)
norm_map  = index_by_frame(norm_files)
flow_map  = index_by_frame(flow_files)

frames = sorted(set(rgb_map) & set(depth_map) & set(norm_map) & set(flow_map))
assert frames, "No overlapping frames across RGB/Depth/Normal/Flow."

print(f"Found {len(frames)} matched frames. First/last: {frames[0]}..{frames[-1]}")

# Preferred channel name sets for your new output nodes
# Depth/Normal/Flow are written via File Output nodes (not Multilayer EXR),
# so channels are typically just R,G,B,A.
DEPTH_PREF  = [("R",), ("Z",), ("Depth",)]
NORMAL_PREF = [("R","G","B"), ("X","Y","Z")]

# IMPORTANT: your flow is packed as u->R and v->A (R,A)
FLOW_PREF   = [("G","R")]

# Read first RGB to set output size
rgb0 = cv2.imread(rgb_map[frames[0]], cv2.IMREAD_COLOR)
if rgb0 is None:
    raise RuntimeError(f"Failed to read {rgb_map[frames[0]]}")
H, W = rgb0.shape[:2]

out_w, out_h = W * 2, H * 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(OUT_MP4, fourcc, FPS, (out_w, out_h))

for k, fr in enumerate(frames):
    rf = rgb_map[fr]
    df = depth_map[fr]
    nf = norm_map[fr]
    ff = flow_map[fr]

    rgb = cv2.imread(rf, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read {rf}")
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    # ---- depth ----
    depth = read_exr_channels(df, DEPTH_PREF)  # (H,W)
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    d_u8 = depth_to_u8(depth, 1, 99)
    d_bgr = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)

    # ---- normals ----
    normals = read_exr_channels(nf, NORMAL_PREF)  # (H,W,3)
    if normals.ndim != 3 or normals.shape[2] < 3:
        raise RuntimeError(f"Normals EXR doesn't look 3-channel: {nf} got shape {normals.shape}")
    normals = normals[..., :3]
    if normals.shape[:2] != (H, W):
        normals = cv2.resize(normals, (W, H), interpolation=cv2.INTER_NEAREST)
    n_bgr = normals_to_bgr_u8(normals)

    # ---- flow ----
    flow = read_exr_channels(ff, FLOW_PREF)  # want (H,W,2) via (R,A)
    if flow.ndim != 3 or flow.shape[2] < 2:
        raise RuntimeError(f"Flow EXR doesn't look 2-channel: {ff} got shape {flow.shape}")
    flow = flow[..., :2]
    if flow.shape[:2] != (H, W):
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_NEAREST)
    f_bgr = flow_vis.flow_to_color(flow, convert_to_bgr=False) #flow_to_bgr_u8(flow, mag_p=99)

    top = np.concatenate([rgb, d_bgr], axis=1)
    bot = np.concatenate([n_bgr, f_bgr], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    vw.write(grid)

    if (k + 1) % 50 == 0:
        print(f"Wrote {k+1}/{len(frames)} frames")

vw.release()
print("Saved:", OUT_MP4)

