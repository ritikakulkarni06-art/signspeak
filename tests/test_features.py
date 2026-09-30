"""Feature extraction tests on synthetic landmarks (no camera or model needed).

Run from the project root: python -m unittest discover -s tests
"""

import unittest

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from src.capture.features import (
    FACE_L_EYEBROW,
    FACE_LIPS,
    FACE_MIRROR,
    FACE_NOSE_BRIDGE,
    FACE_R_EYEBROW,
    FEATURE_DIM,
    LAYOUT,
    POSE_MIRROR,
    extract_features,
    mirror_features,
)
from src.capture.landmarks import FrameLandmarks


def _indices(connections) -> set[int]:
    return {i for c in connections for i in (c.start, c.end)}


def _synthetic_frame(image_size=(1280, 720), seed=0) -> FrameLandmarks:
    rng = np.random.default_rng(seed)
    pose = np.zeros((33, 4), np.float32)
    pose[:, :3] = rng.normal([0.5, 0.6, 0.0], 0.05, (33, 3))
    pose[:, 3] = rng.uniform(0.5, 1.0, 33)
    pose[11, :3] = (0.6, 0.6, 0.0)  # signer's left shoulder
    pose[12, :3] = (0.4, 0.6, 0.0)

    face = rng.normal([0.5, 0.35, 0.0], 0.02, (468, 3)).astype(np.float32)
    face[33] = (0.47, 0.33, 0.0)  # right eye outer corner
    face[263] = (0.53, 0.33, 0.0)  # left eye outer corner
    face[10] = (0.5, 0.27, 0.0)  # forehead
    face[152] = (0.5, 0.43, 0.0)  # chin
    face[FACE_NOSE_BRIDGE] = (0.5, 0.33, 0.0)

    def hand(wrist_x, wrist_y):
        points = rng.normal([wrist_x, wrist_y - 0.05, 0.0], 0.03, (21, 3)).astype(np.float32)
        points[0] = (wrist_x, wrist_y, 0.0)
        return points

    return FrameLandmarks(image_size, pose=pose, left_hand=hand(0.65, 0.7), right_hand=hand(0.35, 0.7), face=face)


def _transform(frame: FrameLandmarks, scale: float, shift: tuple[float, float]) -> FrameLandmarks:
    """Same scene, closer/further from the camera and moved in the image."""

    def apply(points):
        if points is None:
            return None
        out = points.copy()
        out[:, 0] = out[:, 0] * scale + shift[0]
        out[:, 1] = out[:, 1] * scale + shift[1]
        out[:, 2] *= scale
        return out

    return FrameLandmarks(
        frame.image_size, apply(frame.pose), apply(frame.left_hand), apply(frame.right_hand), apply(frame.face)
    )


def _mirror_frame(frame: FrameLandmarks) -> FrameLandmarks:
    """The same scene flipped horizontally: each point takes its left/right
    partner's position, with x flipped. Hands swap as a whole."""

    def flip(points):
        out = points.copy()
        out[:, 0] = 1.0 - out[:, 0]
        return out

    def swap_partners(points, partner):
        flipped = flip(points)
        out = flipped.copy()
        for point, other in partner.items():
            out[point] = flipped[other]
        return out

    return FrameLandmarks(
        frame.image_size,
        pose=swap_partners(frame.pose, POSE_MIRROR),
        left_hand=flip(frame.right_hand),
        right_hand=flip(frame.left_hand),
        face=swap_partners(frame.face, FACE_MIRROR),
    )


class MirrorTests(unittest.TestCase):
    def test_matches_features_of_the_mirrored_scene(self):
        frame = _synthetic_frame()
        np.testing.assert_allclose(
            mirror_features(extract_features(frame)), extract_features(_mirror_frame(frame)), atol=1e-4
        )

    def test_mirroring_twice_is_identity(self):
        features = extract_features(_synthetic_frame(seed=3))
        np.testing.assert_array_equal(mirror_features(mirror_features(features)), features)

    def test_face_partners_map_contours_onto_themselves(self):
        def edges(connections, partner=lambda p: p):
            return {frozenset((partner(c.start), partner(c.end))) for c in connections}

        for contour in (
            FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        ):
            self.assertEqual(edges(contour, lambda p: FACE_MIRROR.get(p, p)), edges(contour))


class FeatureTests(unittest.TestCase):
    def test_region_indices_match_mediapipe(self):
        self.assertEqual(set(FACE_LIPS), _indices(FaceLandmarksConnections.FACE_LANDMARKS_LIPS))
        self.assertEqual(set(FACE_L_EYEBROW), _indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW))
        self.assertEqual(set(FACE_R_EYEBROW), _indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW))
        for region in (FACE_LIPS, FACE_L_EYEBROW, FACE_R_EYEBROW):
            self.assertEqual(len(region), len(set(region)))

    def test_vector_shape_and_flags(self):
        features = extract_features(_synthetic_frame())
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        for part in ("left_hand", "right_hand", "pose", "face"):
            self.assertEqual(features[LAYOUT[f"{part}.present"]][0], 1.0, part)

    def test_invariant_to_camera_distance_and_position(self):
        frame = _synthetic_frame()
        for scale, shift in ((0.5, (0.2, -0.1)), (1.7, (-0.3, 0.05))):
            np.testing.assert_allclose(
                extract_features(_transform(frame, scale, shift)), extract_features(frame), atol=1e-4
            )

    def test_missing_parts_are_zero(self):
        np.testing.assert_array_equal(extract_features(FrameLandmarks((640, 480))), np.zeros(FEATURE_DIM))

        frame = _synthetic_frame()
        face_only = extract_features(FrameLandmarks(frame.image_size, face=frame.face))
        self.assertEqual(face_only[LAYOUT["face.present"]][0], 1.0)
        for name in ("left_hand.present", "left_hand.shape", "pose.present", "face.head_position"):
            self.assertFalse(face_only[LAYOUT[name]].any(), name)

    def test_head_roll_is_separated_from_face_shape(self):
        frame = _synthetic_frame(image_size=(640, 640))
        theta = np.radians(20)
        c, s = np.cos(theta), np.sin(theta)
        center = frame.face[FACE_NOSE_BRIDGE, :2]
        tilted = frame.face.copy()
        tilted[:, :2] = (frame.face[:, :2] - center) @ np.array([[c, s], [-s, c]], np.float32) + center

        upright = extract_features(FrameLandmarks(frame.image_size, face=frame.face))
        rolled = extract_features(FrameLandmarks(frame.image_size, face=tilted))
        self.assertAlmostEqual(upright[LAYOUT["face.head_pose"]][0], 0.0, places=5)
        self.assertAlmostEqual(rolled[LAYOUT["face.head_pose"]][0], theta, places=4)
        for name in ("face.nmm", "face.regions"):
            np.testing.assert_allclose(rolled[LAYOUT[name]], upright[LAYOUT[name]], atol=1e-4)

    def test_mirrored_face_reads_as_upright(self):
        # A mirrored camera swaps which side of the image each eye lands on.
        frame = _synthetic_frame()
        mirrored = frame.face.copy()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        features = extract_features(FrameLandmarks(frame.image_size, face=mirrored))
        np.testing.assert_allclose(features[LAYOUT["face.head_pose"]], 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
