import os

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Static transforms
# ---------------------------------------------------------------------------

# Wrist to wrist camera transform
T_WRIST_WRIST_CAM = np.eye(4)
T_WRIST_WRIST_CAM[:3, :3] = Rotation.from_euler("xyz", [-np.pi / 4, 0, 0]).as_matrix()
T_WRIST_WRIST_CAM[:3, 3] = np.array([0, 0.05, 0])

# 90° CW rotation for head camera display
_R_CW = Rotation.from_euler("z", np.pi / 2).as_matrix()

# Hand skeleton connectivity (21 keypoints)
HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
]


def pose7_to_T(p):
    """Reconstruct 4x4 transform from position (3) + quaternion xyzw (4)."""
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = Rotation.from_quat(p[3:7]).as_matrix()
    return T


class SceneVisualizer:
    """Replay-only scene visualization (cameras + hand keypoints) using Rerun."""

    def __init__(self, app_name="Dataset Replay", spawn=True, memory_limit="50%"):
        os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
        rr.init(app_name)
        if spawn:
            rr.spawn(memory_limit=memory_limit)
        rr.send_blueprint(
            rr.blueprint.Blueprint(
                rr.blueprint.Horizontal(
                    rr.blueprint.Spatial3DView(origin="world", name="3D Scene"),
                    rr.blueprint.Vertical(
                        rr.blueprint.Spatial2DView(
                            origin="world/head_cam/image", name="Head Camera"
                        ),
                        rr.blueprint.Spatial2DView(
                            origin="world/wrist_cam/image", name="Wrist Camera"
                        ),
                    ),
                ),
                collapse_panels=True,
            )
        )
        rr.log(
            "world/world",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )
        self._axes = rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        )

    def log_head_cam(self, T_world_cam, image, focal_length=None):
        rh, rw = image.shape[:2]
        if focal_length is None:
            focal_length = float(max(rw, rh))
        K = np.array([[focal_length, 0, rw / 2], [0, focal_length, rh / 2], [0, 0, 1]])
        rr.log(
            "world/head_cam",
            rr.Transform3D(
                translation=T_world_cam[:3, 3], mat3x3=T_world_cam[:3, :3] @ _R_CW
            ),
        )
        rr.log(
            "world/head_cam/image",
            rr.Pinhole(
                image_from_camera=K, resolution=[rw, rh], image_plane_distance=0.2
            ),
        )
        rr.log("world/head_cam/image", rr.Image(image).compress())
        rr.log("world/head_cam/axes", self._axes)

    def log_wrist_cam(self, T_world_wrist, image):
        T = T_world_wrist @ T_WRIST_WRIST_CAM
        wh, ww = image.shape[:2]
        rr.log(
            "world/wrist_cam",
            rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        )
        rr.log(
            "world/wrist_cam/image",
            rr.Pinhole(
                focal_length=[ww, ww],
                principal_point=[ww / 2, wh / 2],
                resolution=[ww, wh],
                image_plane_distance=0.2,
            ),
        )
        rr.log("world/wrist_cam/image", rr.Image(image).compress())
        rr.log("world/wrist_cam/axes", self._axes)

    def log_hand_keypoints(self, keypoints):
        rr.log(
            "world/hand/keypoints",
            rr.Points3D(keypoints, radii=0.005, colors=[0.8, 0.6, 0.5]),
        )
        segments = np.array([[keypoints[a], keypoints[b]] for a, b in HAND_EDGES])
        rr.log(
            "world/hand/skeleton",
            rr.LineStrips3D(segments, colors=[0.8, 0.6, 0.5]),
        )
