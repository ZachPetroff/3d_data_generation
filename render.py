import bpy
import numpy as np
from mathutils import Matrix
import os
import sys

bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.device = "GPU"
# -----------------------
# User settings
# -----------------------
#PLY_PATH = r"C:\Users\zpetroff\Downloads\mesh_floor.ply"
argv = sys.argv
user_argv = argv[argv.index("--") + 1:] if "--" in argv else []

# Strip Blender/Cycles args you included after `--`
clean = []
i = 0
while i < len(user_argv):
    if user_argv[i] == "--cycles-device":
        i += 2  # skip flag + value (OPTIX/CUDA)
        continue
    clean.append(user_argv[i])
    i += 1

if len(clean) < 2:
    raise SystemExit(
        "Usage: blender -b file.blend -a -- --cycles-device OPTIX <OUT_DIR> <INPUT_PATH>"
    )
print(clean)
OUT_DIR = clean[0]
INPUT_PATH = clean[1]
POS_PATH = os.path.join(INPUT_PATH, "pos.npy")
GAZE_PATH = os.path.join(INPUT_PATH, "gaze.npy")
os.makedirs(OUT_DIR, exist_ok=True)
W, H = 1280, 720
FPS = 30

# If your poses are OpenCV camera convention, this often fixes orientation
CV2BLENDER = Matrix(((1,0,0,0),
                     (0,-1,0,0),
                     (0,0,-1,0),
                     (0,0,0,1)))

APPLY_CV2BLENDER = True  # flip to False if your poses are already Blender-style

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def cam2world_from_pos_fwd_blender(pos, fwd, world_up=np.array([0.0, 0.0, 1.0])):
    """
    pos: (T,3) camera centers in world coordinates
    fwd: (T,3) forward/look direction in world coordinates (doesn't have to be perfectly unit)
    world_up: (3,) chosen global up; usually [0,0,1] (Z-up) or [0,1,0] (Y-up)

    Returns:
      cam2world: (T,4,4) Blender camera-to-world matrices
    """
    pos = np.asarray(pos, dtype=np.float64)
    fwd = normalize(np.asarray(fwd, dtype=np.float64))

    T = pos.shape[0]
    up0 = normalize(np.tile(world_up[None, :], (T, 1)))

    # right = up x forward  (choose this ordering for a right-handed basis)
    right = np.cross(up0, fwd)
    right_norm = np.linalg.norm(right, axis=1)

    # If forward is too close to world_up, cross product gets tiny -> choose a different fallback up
    bad = right_norm < 1e-6
    if np.any(bad):
        alt_up = np.array([0.0, 1.0, 0.0]) if abs(world_up[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        up_alt = normalize(np.tile(alt_up[None, :], (T, 1)))
        right[bad] = np.cross(up_alt[bad], fwd[bad])

    right = normalize(right)
    up = normalize(np.cross(fwd, right))  # completes orthonormal triad

    # Blender: local -Z is forward => local +Z is backward
    back = -fwd

    cam2world = np.zeros((T, 4, 4), dtype=np.float64)
    cam2world[:, 3, 3] = 1.0
    cam2world[:, 0:3, 0] = right
    cam2world[:, 0:3, 1] = up
    cam2world[:, 0:3, 2] = back
    cam2world[:, 0:3, 3] = pos
    return cam2world

def import_ply_any(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PLY not found: {filepath}")

    before = set(bpy.data.objects)

    # Prefer Blender 4.x importer if present
    if hasattr(bpy.ops.wm, "ply_import"):
        res = bpy.ops.wm.ply_import(filepath=filepath)
    elif hasattr(bpy.ops.import_mesh, "ply"):
        res = bpy.ops.import_mesh.ply(filepath=filepath)
    else:
        raise RuntimeError("No PLY import operator found (neither bpy.ops.wm.ply_import nor bpy.ops.import_mesh.ply).")

    after = set(bpy.data.objects)
    new_objs = [o for o in (after - before)]
    mesh_objs = [o for o in new_objs if o.type == "MESH"]

    if not mesh_objs:
        raise RuntimeError(f"PLY import ran but no mesh objects were created. Operator result: {res}")

    # Return the first imported mesh object (plus the full list if you want it)
    return mesh_objs[0], mesh_objs

pos = np.load(POS_PATH)
gaze = np.load(GAZE_PATH)
gaze = -gaze
gaze = gaze / (np.linalg.norm(gaze, axis=1, keepdims=True) + 1e-8) 
poses = cam2world_from_pos_fwd_blender(pos, gaze, world_up=np.array([0,0,1]))

# -----------------------
# Scene setup
# -----------------------
import bpy
import os

# -----------------------
# Scene setup (Blender 4.1)
# -----------------------
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.fps = FPS

# Optional: speed/quality knobs
scene.cycles.samples = 64
# In 4.1 this exists; guard anyway for robustness
if hasattr(scene.cycles, "use_adaptive_sampling"):
    scene.cycles.use_adaptive_sampling = True

# -----------------------
# Import PLY (Blender 4.1: wm.ply_import)
# -----------------------
#if not os.path.isfile(PLY_PATH):
#    raise FileNotFoundError(f"PLY not found: {PLY_PATH}")

#before = set(bpy.data.objects)

# Prefer 4.x importer
#if hasattr(bpy.ops.wm, "ply_import"):
#    bpy.ops.wm.ply_import(filepath=PLY_PATH)
#elif hasattr(bpy.ops.import_mesh, "ply"):  # fallback for older installs
#    bpy.ops.import_mesh.ply(filepath=PLY_PATH)
#else:
#    raise RuntimeError("No PLY import operator found (wm.ply_import or import_mesh.ply).")

#after = set(bpy.data.objects)
#new_objs = list(after - before)
mesh_objs = [o for o in set(bpy.data.objects) if o.type == "MESH"]

if not mesh_objs:
    raise RuntimeError("PLY import ran but did not create any MESH objects.")

ply_obj = mesh_objs[0]
mesh = ply_obj.data

# -----------------------
# Find vertex color attribute (Blender 4.1)
# -----------------------
vcol_name = None
if hasattr(mesh, "color_attributes") and mesh.color_attributes:
    # Prefer render-active, then active, then first
    render_active = getattr(mesh.color_attributes, "active_render", None)
    active = getattr(mesh.color_attributes, "active_color", None)

    if render_active is not None:
        vcol_name = render_active.name
    elif active is not None:
        vcol_name = active.name
    else:
        vcol_name = mesh.color_attributes[0].name

# Fallback guesses
if vcol_name is None and hasattr(mesh, "color_attributes"):
    for guess in ("Col", "Color", "color", "vcol", "vertex_colors"):
        try:
            _ = mesh.color_attributes[guess]
            vcol_name = guess
            break
        except KeyError:
            pass

print("Using vertex color attribute:", vcol_name)

# -----------------------
# Material: use Color Attribute node (preferred in 4.x)
# -----------------------
mat = bpy.data.materials.new(name="PLY_VertexColor")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

out = nodes.new(type="ShaderNodeOutputMaterial")
bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
out.location = (400, 0)
bsdf.location = (0, 0)
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

if vcol_name is not None:
    # Blender 4.x has a dedicated Color Attribute node
    try:
        colattr = nodes.new(type="ShaderNodeVertexColor")  # exists in some builds
        # In 4.x this node often exposes "layer_name"
        if hasattr(colattr, "layer_name"):
            colattr.layer_name = vcol_name
        else:
            # fallback to attribute_name if present
            if hasattr(colattr, "attribute_name"):
                colattr.attribute_name = vcol_name
        colattr.location = (-300, 0)
        links.new(colattr.outputs["Color"], bsdf.inputs["Base Color"])
    except RuntimeError:
        # Fallback: Attribute node works for color attributes too
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.location = (-300, 0)
        attr.attribute_name = vcol_name
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
else:
    print("WARNING: No vertex color attribute found. PLY will render with default material color.")

# Assign material to the object
if ply_obj.data.materials:
    ply_obj.data.materials[0] = mat
else:
    ply_obj.data.materials.append(mat)


# -----------------------
# Create camera
# -----------------------
bpy.ops.object.camera_add()
cam = bpy.context.active_object
scene.camera = cam
cam.animation_data_clear()
cam.rotation_mode = 'XYZ'

# -----------------------
# Camera parameters (FOV / lens)
# -----------------------
cam.data.lens_unit = 'MILLIMETERS'     # or 'FOV'
cam.data.lens = 18.0                  # smaller -> wider (e.g., 35, 24, 18, 14)

# Optional: sensor size (defaults ~36mm width)
cam.data.sensor_width = 36.0          # "full frame" horizontal sensor
cam.data.sensor_height = 24.0

cam.data.sensor_fit = 'HORIZONTAL'    # keep horizontal FOV stable with your W,H

# Depth-relevant: clipping range (set to cover your scene)
cam.data.clip_start = 0.01
cam.data.clip_end = 1000.0

# -----------------------
# poses
# -----------------------
N = poses.shape[0]
scene.frame_start = 1
scene.frame_end = N

for i in range(N):
    M = Matrix(poses[i].tolist())  # cam2world
    if APPLY_CV2BLENDER:
        M = M @ CV2BLENDER
    
    for r in range(3):
        M[r][0] *= -1  # flip X column
        M[r][1] *= -1

    cam.matrix_world = M
    cam.keyframe_insert(data_path="location", frame=i+1)
    cam.keyframe_insert(data_path="rotation_euler", frame=i+1)
    
action = cam.animation_data.action
for fcurve in action.fcurves:
    for kp in fcurve.keyframe_points:
        kp.interpolation = 'LINEAR'

scene = bpy.context.scene
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

# Passes
vl = bpy.context.view_layer
vl.use_pass_z = True
vl.use_pass_normal = True
vl.use_pass_vector = True
vl.use_motion_blur = False
scene.render.use_motion_blur = False

rlayers = tree.nodes.new("CompositorNodeRLayers")
rlayers.location = (0, 0)

# -----------------------
# PNG: RGB
# -----------------------
file_out_png = tree.nodes.new("CompositorNodeOutputFile")
file_out_png.base_path = OUT_DIR
file_out_png.location = (450, 200)

file_out_png.format.file_format = "PNG"
file_out_png.format.color_mode = "RGB"
file_out_png.format.color_depth = "8"

# PNG: RGB
file_out_png.file_slots.clear()
file_out_png.file_slots.new("RGB")
file_out_png.file_slots[0].path = "####_RGB"  # Access by index

tree.links.new(rlayers.outputs["Image"], file_out_png.inputs["RGB"])

# -----------------------
# EXR: Depth, Normal, Optic Flow
# -----------------------
file_out_exr = tree.nodes.new("CompositorNodeOutputFile")
file_out_exr.base_path = OUT_DIR
file_out_exr.location = (450, -150)

file_out_exr.format.file_format = "OPEN_EXR"
file_out_exr.format.color_depth = "32"
file_out_exr.format.color_mode = "RGBA"

# EXR: Depth, Normal, Optic Flow
file_out_exr.file_slots.clear()
file_out_exr.file_slots.new("Depth")
file_out_exr.file_slots.new("Normal")
file_out_exr.file_slots.new("Optic_Flow")

file_out_exr.file_slots[0].path = "####_Depth"
file_out_exr.file_slots[1].path = "####_Normal"
file_out_exr.file_slots[2].path = "####_Optic_Flow"

tree.links.new(rlayers.outputs["Depth"],  file_out_exr.inputs["Depth"])
tree.links.new(rlayers.outputs["Normal"], file_out_exr.inputs["Normal"])

# Vector -> Separate -> Combine (B->R, A->A)
sep = tree.nodes.new("CompositorNodeSepRGBA")
sep.location = (200, -420)
comb = tree.nodes.new("CompositorNodeCombRGBA")
comb.location = (350, -420)

comb.inputs["G"].default_value = 0.0
comb.inputs["B"].default_value = 0.0

tree.links.new(rlayers.outputs["Vector"], sep.inputs["Image"])
tree.links.new(sep.outputs["B"], comb.inputs["R"])
tree.links.new(sep.outputs["A"], comb.inputs["A"])
tree.links.new(comb.outputs["Image"], file_out_exr.inputs["Optic_Flow"])

# -----------------------
# Render
# -----------------------
bpy.ops.render.render(animation=True)
