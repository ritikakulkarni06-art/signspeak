"""Webcam capture and MediaPipe Holistic landmarks -> one feature vector per frame."""

from .camera import open_webcam
from .features import FEATURE_DIM, LAYOUT, NMM_NAMES, extract_features, mirror_features
from .holistic import HolisticTracker
from .landmarks import FrameLandmarks
from .overlay import draw_landmarks
from .segmenter import SegmenterConfig, SignSegment, SignSegmenter, segment_signs

__all__ = [
    "FEATURE_DIM",
    "LAYOUT",
    "NMM_NAMES",
    "FrameLandmarks",
    "HolisticTracker",
    "SegmenterConfig",
    "SignSegment",
    "SignSegmenter",
    "draw_landmarks",
    "extract_features",
    "mirror_features",
    "open_webcam",
    "segment_signs",
]
