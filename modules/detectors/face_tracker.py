"""
Face Tracking Module

Implements multi-object tracking using ByteTrack algorithm to assign consistent IDs
to faces across video frames.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

import config
from modules.detectors.yolo_detector import FaceDetection


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """
    Container for tracked face with persistent ID.

    Attributes:
        track_id: Unique identifier for this face across frames
        detection: Latest face detection
        age: Number of frames since last detection
        hits: Total number of successful detections
        history: Recent bounding box positions
    """
    track_id: int
    detection: FaceDetection
    age: int = 0
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self):
        """Initialize history with current detection."""
        self.history.append(self.detection.bbox)


class SimpleFaceTracker:
    """
    Simple face tracker using IoU-based association.

    Assigns unique IDs to detected faces and tracks them across frames.
    Uses Intersection over Union (IoU) for matching detections to existing tracks.
    """

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        min_hits: int = config.TRACKER_MIN_HITS,
        iou_threshold: float = config.TRACKER_IOU_THRESHOLD
    ):
        """
        Initialize face tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[TrackedFace] = []
        self.next_id = 1
        self.frame_count = 0

        logger.info(f"SimpleFaceTracker initialized: max_age={max_age}, "
                   f"min_hits={min_hits}, iou={iou_threshold}")

    def update(self, detections: List[FaceDetection]) -> List[TrackedFace]:
        """
        Update tracks with new detections from current frame.

        Args:
            detections: List of face detections from current frame

        Returns:
            List of confirmed tracked faces

        Example:
            >>> tracker = SimpleFaceTracker()
            >>> detections = detector.detect_faces(frame)
            >>> tracked_faces = tracker.update(detections)
        """
        self.frame_count += 1

        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections(detections)

        # Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            self.tracks[track_idx].detection = detections[detection_idx]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].age = 0
            self.tracks[track_idx].history.append(detections[detection_idx].bbox)

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = TrackedFace(
                track_id=self.next_id,
                detection=detections[detection_idx]
            )
            self.tracks.append(new_track)
            self.next_id += 1

        # Increment age for tracks without matches
        for track in self.tracks:
            if track.age > 0 or track.track_id not in [
                t.track_id for t_idx, _ in matched_tracks for t in [self.tracks[t_idx]]
            ]:
                track.age += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Return only confirmed tracks
        confirmed_tracks = [
            t for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]

        logger.debug(f"Frame {self.frame_count}: {len(confirmed_tracks)} tracked faces, "
                    f"{len(self.tracks)} total tracks")

        return confirmed_tracks

    def _match_detections(
        self,
        detections: List[FaceDetection]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Match detections to existing tracks using IoU.

        Args:
            detections: List of new detections

        Returns:
            Tuple of (matched_pairs, unmatched_detection_indices)
            matched_pairs: List of (track_idx, detection_idx) tuples
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t_idx, track in enumerate(self.tracks):
            for d_idx, detection in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(
                    track.detection.bbox,
                    detection.bbox
                )

        # Greedy matching (highest IoU first)
        matched_tracks = []
        matched_detections = set()
        matched_track_indices = set()

        # Sort by IoU value (descending)
        iou_pairs = []
        for t_idx in range(len(self.tracks)):
            for d_idx in range(len(detections)):
                iou_pairs.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))

        iou_pairs.sort(reverse=True)

        # Match greedily
        for iou, t_idx, d_idx in iou_pairs:
            if iou < self.iou_threshold:
                break

            if t_idx not in matched_track_indices and d_idx not in matched_detections:
                matched_tracks.append((t_idx, d_idx))
                matched_track_indices.add(t_idx)
                matched_detections.add(d_idx)

        # Find unmatched detections
        unmatched_detections = [
            d_idx for d_idx in range(len(detections))
            if d_idx not in matched_detections
        ]

        return matched_tracks, unmatched_detections

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bbox (x1, y1, x2, y2)
            bbox2: Second bbox (x1, y1, x2, y2)

        Returns:
            IoU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def get_track_by_id(self, track_id: int) -> Optional[TrackedFace]:
        """
        Get track by its ID.

        Args:
            track_id: Track ID to search for

        Returns:
            TrackedFace if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_active_tracks(self) -> List[TrackedFace]:
        """
        Get all currently active (confirmed) tracks.

        Returns:
            List of confirmed tracked faces
        """
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age == 0]

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking stats
        """
        active_tracks = self.get_active_tracks()

        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active_tracks),
            "next_id": self.next_id,
            "frame_count": self.frame_count,
            "avg_track_age": np.mean([t.age for t in self.tracks]) if self.tracks else 0,
            "avg_track_hits": np.mean([t.hits for t in self.tracks]) if self.tracks else 0
        }
