"""
Video Frame Extraction Module

Handles video loading, validation, and frame extraction with timestamps.
Implements efficient frame-by-frame processing with configurable sampling rates.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import config


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """
    Container for video metadata information.

    Attributes:
        file_path: Path to video file
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        total_frames: Total number of frames
        duration: Video duration in seconds
        codec: Video codec used
        file_size_mb: File size in megabytes
    """
    file_path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    file_size_mb: float


class VideoFrameExtractor:
    """
    Extracts frames from video files with configurable sampling rates.

    Supports efficient frame-by-frame iteration with timestamps,
    video validation, and automatic resource cleanup.
    """

    def __init__(self, video_path: str | Path):
        """
        Initialize video frame extractor.

        Args:
            video_path: Path to video file

        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If video format is not supported or file is too large
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Видео файл не найден: {self.video_path}")

        # Validate video file
        self._validate_video()

        # Initialize capture object (will be created when needed)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

        logger.info(f"VideoFrameExtractor initialized for: {self.video_path.name}")

    def _validate_video(self) -> None:
        """
        Validate video file format and size.

        Raises:
            ValueError: If video format is unsupported or file is too large
        """
        # Check file extension
        if self.video_path.suffix.lower() not in config.SUPPORTED_VIDEO_FORMATS:
            error_msg = config.ERROR_MESSAGES_RU["video_format_unsupported"]
            logger.error(f"Unsupported video format: {self.video_path.suffix}")
            raise ValueError(error_msg)

        # Check file size
        file_size_mb = self.video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.MAX_VIDEO_SIZE_MB:
            error_msg = config.ERROR_MESSAGES_RU["video_too_large"]
            logger.error(f"Video too large: {file_size_mb:.2f} MB")
            raise ValueError(error_msg)

        logger.debug(f"Video validation passed: {file_size_mb:.2f} MB")

    def _initialize_capture(self) -> cv2.VideoCapture:
        """
        Initialize OpenCV VideoCapture object.

        Returns:
            OpenCV VideoCapture object

        Raises:
            RuntimeError: If video cannot be opened
        """
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            error_msg = config.ERROR_MESSAGES_RU["video_load_failed"]
            logger.error(f"Failed to open video: {self.video_path}")
            raise RuntimeError(error_msg)

        return cap

    def get_metadata(self) -> VideoMetadata:
        """
        Extract and return video metadata.

        Returns:
            VideoMetadata object with video information

        Raises:
            RuntimeError: If video cannot be opened
        """
        if self._metadata is not None:
            return self._metadata

        cap = self._initialize_capture()

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            file_size_mb = self.video_path.stat().st_size / (1024 * 1024)

            self._metadata = VideoMetadata(
                file_path=self.video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec,
                file_size_mb=file_size_mb
            )

            logger.info(f"Video metadata: {width}x{height}, {fps:.2f} FPS, "
                       f"{total_frames} frames, {duration:.2f}s")

            return self._metadata

        finally:
            cap.release()

    def extract_frames(
        self,
        sample_rate: int = 1,
        resize_width: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames from video as a generator.

        This is a memory-efficient method that yields frames one at a time
        rather than loading all frames into memory.

        Args:
            sample_rate: Process every Nth frame (default: 1, process all frames)
            resize_width: Resize frame to this width, maintaining aspect ratio
            start_frame: Frame number to start from (default: 0)
            end_frame: Frame number to end at (default: None, process until end)

        Yields:
            Tuple of (frame_number, timestamp_seconds, frame_array)

        Raises:
            RuntimeError: If video cannot be opened or read

        Example:
            >>> extractor = VideoFrameExtractor("video.mp4")
            >>> for frame_num, timestamp, frame in extractor.extract_frames(sample_rate=5):
            ...     # Process every 5th frame
            ...     process_frame(frame)
        """
        cap = self._initialize_capture()
        metadata = self.get_metadata()

        try:
            frame_count = 0
            processed_count = 0

            # Set start position if specified
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_count = start_frame

            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.debug(f"Finished reading video at frame {frame_count}")
                    break

                # Check if we've reached end frame
                if end_frame is not None and frame_count >= end_frame:
                    logger.debug(f"Reached end frame {end_frame}")
                    break

                # Process only every Nth frame based on sample rate
                if frame_count % sample_rate == 0:
                    # Calculate timestamp
                    timestamp = frame_count / metadata.fps if metadata.fps > 0 else 0

                    # Resize if requested
                    if resize_width is not None and resize_width != metadata.width:
                        aspect_ratio = metadata.height / metadata.width
                        new_height = int(resize_width * aspect_ratio)
                        frame = cv2.resize(frame, (resize_width, new_height))

                    yield frame_count, timestamp, frame
                    processed_count += 1

                frame_count += 1

            logger.info(f"Extracted {processed_count} frames from {frame_count} total frames")

        except Exception as e:
            logger.error(f"Error during frame extraction: {str(e)}")
            raise RuntimeError(f"Ошибка при извлечении кадров: {str(e)}")

        finally:
            cap.release()

    def extract_frame_at_timestamp(
        self,
        timestamp: float,
        resize_width: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific timestamp.

        Args:
            timestamp: Time in seconds
            resize_width: Resize frame to this width, maintaining aspect ratio

        Returns:
            Frame as numpy array, or None if timestamp is invalid

        Raises:
            RuntimeError: If video cannot be opened
        """
        metadata = self.get_metadata()

        if timestamp < 0 or timestamp > metadata.duration:
            logger.warning(f"Timestamp {timestamp}s out of range [0, {metadata.duration}s]")
            return None

        cap = self._initialize_capture()

        try:
            # Set position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            ret, frame = cap.read()

            if not ret:
                logger.error(f"Failed to read frame at timestamp {timestamp}s")
                return None

            # Resize if requested
            if resize_width is not None and resize_width != metadata.width:
                aspect_ratio = metadata.height / metadata.width
                new_height = int(resize_width * aspect_ratio)
                frame = cv2.resize(frame, (resize_width, new_height))

            logger.debug(f"Extracted frame at timestamp {timestamp}s")
            return frame

        finally:
            cap.release()

    def extract_frame_at_number(
        self,
        frame_number: int,
        resize_width: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific frame number.

        Args:
            frame_number: Frame index (0-based)
            resize_width: Resize frame to this width, maintaining aspect ratio

        Returns:
            Frame as numpy array, or None if frame number is invalid

        Raises:
            RuntimeError: If video cannot be opened
        """
        metadata = self.get_metadata()

        if frame_number < 0 or frame_number >= metadata.total_frames:
            logger.warning(f"Frame number {frame_number} out of range "
                         f"[0, {metadata.total_frames})")
            return None

        cap = self._initialize_capture()

        try:
            # Set position to frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()

            if not ret:
                logger.error(f"Failed to read frame {frame_number}")
                return None

            # Resize if requested
            if resize_width is not None and resize_width != metadata.width:
                aspect_ratio = metadata.height / metadata.width
                new_height = int(resize_width * aspect_ratio)
                frame = cv2.resize(frame, (resize_width, new_height))

            logger.debug(f"Extracted frame number {frame_number}")
            return frame

        finally:
            cap.release()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of video information as a dictionary.

        Returns:
            Dictionary with video metadata
        """
        metadata = self.get_metadata()

        return {
            "file_name": metadata.file_path.name,
            "file_path": str(metadata.file_path),
            "resolution": f"{metadata.width}x{metadata.height}",
            "fps": metadata.fps,
            "total_frames": metadata.total_frames,
            "duration_seconds": metadata.duration,
            "duration_formatted": format_duration(metadata.duration),
            "codec": metadata.codec,
            "file_size_mb": round(metadata.file_size_mb, 2)
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._cap is not None:
            self._cap.release()
        logger.debug("VideoFrameExtractor resources cleaned up")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1:23:45" or "5:30")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def validate_video_path(video_path: str | Path) -> bool:
    """
    Quick validation of video file without full initialization.

    Args:
        video_path: Path to video file

    Returns:
        True if video is valid, False otherwise
    """
    try:
        path = Path(video_path)

        # Check existence
        if not path.exists():
            return False

        # Check format
        if path.suffix.lower() not in config.SUPPORTED_VIDEO_FORMATS:
            return False

        # Check size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.MAX_VIDEO_SIZE_MB:
            return False

        # Try to open with OpenCV
        cap = cv2.VideoCapture(str(path))
        is_valid = cap.isOpened()
        cap.release()

        return is_valid

    except Exception as e:
        logger.error(f"Video validation error: {str(e)}")
        return False
