"""Prova dal vivo: scontornatura da immagine + pianificazione FM 7000."""

from fm7000.live.planner import LivePlan, LivePlanner, detection_to_slice
from fm7000.live.segmentation import SliceDetection2D, segment_frame

__all__ = [
    "LivePlan",
    "LivePlanner",
    "SliceDetection2D",
    "detection_to_slice",
    "segment_frame",
]
