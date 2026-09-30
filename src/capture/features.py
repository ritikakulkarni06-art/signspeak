"""Turn one frame of Holistic landmarks into a fixed-length feature vector.

Layout (float32, FEATURE_DIM values; LAYOUT maps each name to its slice):

    left_hand.present      1   1 if the hand was tracked this frame
    left_hand.location     3   wrist x, y in body frame + hand size / shoulder width
    left_hand.shape       63   21 points x (x, y, z) in hand frame
    right_hand.*               same as left_hand
    pose.present           1
    pose.points           21   nose, shoulders, elbows, wrists x (x, y, z) in body frame
    pose.visibility        7   MediaPipe visibility for those points
    face.present           1
    face.head_pose         3   roll, pitch, yaw in radians
    face.head_position     3   nose bridge x, y in body frame + eye distance / shoulder width
    face.nmm               6   non-manual marker scalars, see NMM_NAMES
    face.regions         180   eyebrows (20) + lips (40) x (x, y, z) in head frame

A part that wasn't tracked is all zeros with present = 0. Location features
(hand/head position) also stay zero when the pose is missing, because they are
measured in the body frame.

Normalization:
  * MediaPipe divides x by image width and y by height. x and z are rescaled by
    width / height first so a unit means the same distance in every direction.
  * Body frame: origin at the shoulder midpoint, unit = shoulder width (x, y
    only; pose z is too noisy at webcam range). Removes where the signer stands
    and how far they are from the camera.
  * Hand frame: origin at the wrist, unit = mean wrist-to-knuckle distance.
    Removes hand size and distance. Orientation is kept on purpose: palm
    orientation is part of a sign.
  * Head frame: origin at the nose bridge, rotated onto the head's own axes,
    unit = outer eye-corner distance. Eyebrow and mouth shape are therefore
    independent of head tilt, which is reported separately in face.head_pose.

The head frame's x axis always points toward image-right, whichever eye
MediaPipe puts there (it depends on whether the camera mirrors), so an upright
head is always roll = 0 rather than wrapping to +-180 degrees.

Head pose signs, in image terms (approximate): roll > 0 means the image-right
eye is lower, pitch > 0 means the chin is tipped up, yaw > 0 means the head is
turned toward image-right.
"""

from __future__ import annotations

import numpy as np

from .landmarks import FrameLandmarks

# MediaPipe Pose indices (33 points).
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_POINTS = (0, 11, 12, 13, 14, 15, 16)  # nose, shoulders, elbows, wrists

# MediaPipe Hand indices (21 points).
HAND_POINTS = 21
HAND_WRIST = 0
HAND_KNUCKLES = (5, 9, 13, 17)  # index, middle, ring, pinky MCP joints

# MediaPipe Face Mesh indices (478 points; only the first 468 are used). The region lists match
# FaceLandmarksConnections; they're hardcoded so the feature layout can't
# shift under a library upgrade (tests check them against MediaPipe).
FACE_NOSE_BRIDGE, FACE_FOREHEAD, FACE_CHIN = 168, 10, 152
FACE_R_EYE_OUTER, FACE_R_EYE_INNER, FACE_R_EYE_TOP, FACE_R_EYE_BOTTOM = 33, 133, 159, 145
FACE_L_EYE_OUTER, FACE_L_EYE_INNER, FACE_L_EYE_TOP, FACE_L_EYE_BOTTOM = 263, 362, 386, 374
FACE_MOUTH_RIGHT, FACE_MOUTH_LEFT = 61, 291
FACE_LIP_UPPER_INNER, FACE_LIP_LOWER_INNER = 13, 14
FACE_R_EYEBROW = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
FACE_L_EYEBROW = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
FACE_LIPS = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)
FACE_REGION_POINTS = FACE_R_EYEBROW + FACE_L_EYEBROW + FACE_LIPS

NMM_NAMES = (
    "brow_raise_right",  # eye center to eyebrow height
    "brow_raise_left",
    "eye_open_right",  # upper to lower eyelid
    "eye_open_left",
    "mouth_open",  # inner upper to inner lower lip
    "mouth_width",  # corner to corner
)

_HAND_SECTIONS = (("present", 1), ("location", 3), ("shape", HAND_POINTS * 3))
_SECTIONS = (
    *((f"left_hand.{name}", size) for name, size in _HAND_SECTIONS),
    *((f"right_hand.{name}", size) for name, size in _HAND_SECTIONS),
    ("pose.present", 1),
    ("pose.points", len(POSE_POINTS) * 3),
    ("pose.visibility", len(POSE_POINTS)),
    ("face.present", 1),
    ("face.head_pose", 3),
    ("face.head_position", 3),
    ("face.nmm", len(NMM_NAMES)),
    ("face.regions", len(FACE_REGION_POINTS) * 3),
)

LAYOUT: dict[str, slice] = {}
_offset = 0
for _name, _size in _SECTIONS:
    LAYOUT[_name] = slice(_offset, _offset + _size)
    _offset += _size
FEATURE_DIM = _offset

# Signer's right <-> left partner of each used point, for mirroring. The eyebrow
# tuples are ordered so FACE_R_EYEBROW[i] mirrors FACE_L_EYEBROW[i]; the lip
# pairs follow the lip contours. Points on the midline map to themselves.
_FACE_MIRROR_PAIRS = (
    (FACE_R_EYE_OUTER, FACE_L_EYE_OUTER),
    (FACE_R_EYE_INNER, FACE_L_EYE_INNER),
    (FACE_R_EYE_TOP, FACE_L_EYE_TOP),
    (FACE_R_EYE_BOTTOM, FACE_L_EYE_BOTTOM),
    *zip(FACE_R_EYEBROW, FACE_L_EYEBROW),
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (185, 409), (40, 270), (39, 269), (37, 267),
    (78, 308), (95, 324), (88, 318), (178, 402), (87, 317), (191, 415), (80, 310), (81, 311), (82, 312),
)  # fmt: skip
FACE_MIRROR = {**dict(_FACE_MIRROR_PAIRS), **{left: right for right, left in _FACE_MIRROR_PAIRS}}
POSE_MIRROR = {0: 0, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
_NMM_MIRROR = {
    "brow_raise_right": "brow_raise_left",
    "brow_raise_left": "brow_raise_right",
    "eye_open_right": "eye_open_left",
    "eye_open_left": "eye_open_right",
}


def _mirror_tables() -> tuple[np.ndarray, np.ndarray]:
    """Mirroring the scene is a signed permutation of the vector: output[i] =
    input[source[i]] * sign[i]. x coordinates flip sign, left/right swap."""
    source = np.arange(FEATURE_DIM)
    sign = np.ones(FEATURE_DIM, np.float32)

    def slots(name: str) -> np.ndarray:
        return np.arange(FEATURE_DIM)[LAYOUT[name]]

    def mirror_points(name: str, order: tuple[int, ...], partner: dict[int, int]) -> None:
        points = slots(name).reshape(-1, 3)
        for k, point in enumerate(order):
            source[points[k]] = points[order.index(partner.get(point, point))]
            sign[points[k][0]] = -1.0

    for hand, other in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
        for part in ("present", "location", "shape"):
            source[slots(f"{hand}.{part}")] = slots(f"{other}.{part}")
        sign[slots(f"{hand}.location")[0]] = -1.0
        sign[slots(f"{hand}.shape")[0::3]] = -1.0

    mirror_points("pose.points", POSE_POINTS, POSE_MIRROR)
    visibility = slots("pose.visibility")
    source[visibility] = visibility[[POSE_POINTS.index(POSE_MIRROR[p]) for p in POSE_POINTS]]

    roll, _, yaw = slots("face.head_pose")
    sign[[roll, yaw]] = -1.0
    sign[slots("face.head_position")[0]] = -1.0
    nmm = slots("face.nmm")
    source[nmm] = nmm[[NMM_NAMES.index(_NMM_MIRROR.get(name, name)) for name in NMM_NAMES]]
    mirror_points("face.regions", FACE_REGION_POINTS, FACE_MIRROR)
    return source, sign


_MIRROR_SOURCE, _MIRROR_SIGN = _mirror_tables()


def mirror_features(features: np.ndarray) -> np.ndarray:
    """Features of the horizontally mirrored scene, where the signer's left and
    right swap. Works on (..., FEATURE_DIM). Used to put every signer's dominant
    hand in the same slot; also usable as data augmentation."""
    return features[..., _MIRROR_SOURCE] * _MIRROR_SIGN

_EPS = 1e-6
_POSE_IDX = list(POSE_POINTS)
_KNUCKLE_IDX = list(HAND_KNUCKLES)
_R_EYE_IDX = [FACE_R_EYE_OUTER, FACE_R_EYE_INNER, FACE_R_EYE_TOP, FACE_R_EYE_BOTTOM]
_L_EYE_IDX = [FACE_L_EYE_OUTER, FACE_L_EYE_INNER, FACE_L_EYE_TOP, FACE_L_EYE_BOTTOM]
_R_BROW_IDX = list(FACE_R_EYEBROW)
_L_BROW_IDX = list(FACE_L_EYEBROW)
_REGION_IDX = list(FACE_REGION_POINTS)


def extract_features(frame: FrameLandmarks) -> np.ndarray:
    """Return the FEATURE_DIM float32 feature vector for one frame."""
    out = np.zeros(FEATURE_DIM, np.float32)
    aspect = frame.image_size[0] / frame.image_size[1]

    body = None
    if frame.pose is not None:
        pose = _isotropic(frame.pose[:, :3], aspect)
        body = _body_frame(pose)
        if body is not None:
            origin, unit = body
            out[LAYOUT["pose.present"]] = 1.0
            out[LAYOUT["pose.points"]] = ((pose[_POSE_IDX] - origin) / unit).ravel()
            out[LAYOUT["pose.visibility"]] = frame.pose[_POSE_IDX, 3]

    for side, hand in (("left_hand", frame.left_hand), ("right_hand", frame.right_hand)):
        if hand is not None:
            _write_hand(out, side, _isotropic(hand, aspect), body)

    if frame.face is not None:
        _write_face(out, _isotropic(frame.face, aspect), body)
    return out


def _isotropic(points: np.ndarray, aspect: float) -> np.ndarray:
    return points * np.array([aspect, 1.0, aspect], np.float32)


def _body_frame(pose: np.ndarray) -> tuple[np.ndarray, float] | None:
    left, right = pose[POSE_L_SHOULDER], pose[POSE_R_SHOULDER]
    unit = float(np.linalg.norm((left - right)[:2]))
    if unit < _EPS:
        return None
    return (left + right) / 2, unit


def _write_hand(out: np.ndarray, side: str, hand: np.ndarray, body) -> None:
    wrist = hand[HAND_WRIST]
    size = float(np.linalg.norm(hand[_KNUCKLE_IDX] - wrist, axis=1).mean())
    if size < _EPS:
        return
    out[LAYOUT[f"{side}.present"]] = 1.0
    out[LAYOUT[f"{side}.shape"]] = ((hand - wrist) / size).ravel()
    if body is not None:
        origin, unit = body
        out[LAYOUT[f"{side}.location"]] = (*((wrist[:2] - origin[:2]) / unit), size / unit)


def _write_face(out: np.ndarray, face: np.ndarray, body) -> None:
    across = face[FACE_L_EYE_OUTER] - face[FACE_R_EYE_OUTER]
    eye_dist = float(np.linalg.norm(across))
    if eye_dist < _EPS:
        return
    x_axis = across / eye_dist
    if x_axis[0] < 0:
        x_axis = -x_axis
    down = face[FACE_CHIN] - face[FACE_FOREHEAD]
    y_axis = down - x_axis * (down @ x_axis)
    y_len = float(np.linalg.norm(y_axis))
    if y_len < _EPS:
        return
    y_axis = y_axis / y_len
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns: head axes in image space
    head = (face - face[FACE_NOSE_BRIDGE]) @ rotation / eye_dist

    out[LAYOUT["face.present"]] = 1.0
    out[LAYOUT["face.head_pose"]] = (
        np.arctan2(x_axis[1], x_axis[0]),  # roll
        np.arcsin(np.clip(-y_axis[2], -1.0, 1.0)),  # pitch
        np.arcsin(np.clip(x_axis[2], -1.0, 1.0)),  # yaw
    )
    if body is not None:
        origin, unit = body
        nose = face[FACE_NOSE_BRIDGE]
        out[LAYOUT["face.head_position"]] = (*((nose[:2] - origin[:2]) / unit), eye_dist / unit)
    out[LAYOUT["face.nmm"]] = _non_manual_markers(head)
    out[LAYOUT["face.regions"]] = head[_REGION_IDX].ravel()


def _non_manual_markers(head: np.ndarray) -> tuple[float, ...]:
    # Head frame y points down (toward the chin), so "above" means smaller y.
    return (
        head[_R_EYE_IDX, 1].mean() - head[_R_BROW_IDX, 1].mean(),
        head[_L_EYE_IDX, 1].mean() - head[_L_BROW_IDX, 1].mean(),
        head[FACE_R_EYE_BOTTOM, 1] - head[FACE_R_EYE_TOP, 1],
        head[FACE_L_EYE_BOTTOM, 1] - head[FACE_L_EYE_TOP, 1],
        head[FACE_LIP_LOWER_INNER, 1] - head[FACE_LIP_UPPER_INNER, 1],
        float(np.linalg.norm(head[FACE_MOUTH_LEFT] - head[FACE_MOUTH_RIGHT])),
    )
