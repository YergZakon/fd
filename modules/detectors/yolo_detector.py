"""
YOLOv11 Face Detection Module

Implements face detection using YOLOv11 from Ultralytics.
Provides efficient face detection with confidence scoring and bounding boxes.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")

import config


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """
    Container for face detection results.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Detection confidence score (0-1)
        class_id: Class ID from model (typically 0 for person/face)
        face_crop: Cropped face image (optional)
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    face_crop: Optional[np.ndarray] = None

    def get_bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box in xyxy format."""
        return self.bbox

    def get_bbox_xywh(self) -> Tuple[int, int, int, int]:
        """Get bounding box in xywh format."""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    def get_area(self) -> int:
        """Get area of bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class YOLOFaceDetector:
    """
    Face detector using YOLOv11 model.

    Handles model loading, inference, and post-processing of detections.
    Supports both standard YOLO models and fine-tuned face detection models.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.YOLO_IOU_THRESHOLD,
        device: str = config.YOLO_DEVICE
    ):
        """
        Initialize YOLO face detector.

        Args:
            model_path: Path to YOLO model weights (if None, uses default from config)
            confidence_threshold: Minimum confidence for detection (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'mps')

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model cannot be loaded
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Determine model path
        if model_path is None:
            self.model_path = config.YOLO_MODEL_PATH
        else:
            self.model_path = Path(model_path)

        # Load model
        self.model = self._load_model()

        logger.info(f"YOLOFaceDetector initialized with {self.model_path.name} "
                   f"on device: {self.device}")

    def _load_model(self) -> YOLO:
        """
        Load YOLO model from file or download if not exists.

        Returns:
            Loaded YOLO model

        Raises:
            RuntimeError: If model cannot be loaded
        """
        try:
            # If model doesn't exist locally, download it using just the model name
            if not self.model_path.exists():
                logger.info(f"Model not found locally. Downloading {config.YOLO_MODEL_NAME}...")
                config.MODELS_DIR.mkdir(exist_ok=True)

                # Download model using just the name (YOLO handles download)
                model = YOLO(config.YOLO_MODEL_NAME)

                # Save to our models directory
                import shutil
                downloaded_path = Path.home() / '.cache' / 'ultralytics' / config.YOLO_MODEL_NAME
                if downloaded_path.exists():
                    shutil.copy(downloaded_path, self.model_path)
                    logger.info(f"Model saved to: {self.model_path}")
            else:
                # Load existing model
                model = YOLO(str(self.model_path))

            # Set device and optimize for GPU
            model.to(self.device)

            # Log GPU optimization info
            if self.device == "cuda":
                logger.info(f"GPU acceleration enabled on {self.device}")
                if hasattr(config, 'YOLO_USE_HALF_PRECISION') and config.YOLO_USE_HALF_PRECISION:
                    logger.info("FP16 (half precision) will be used for inference")

            logger.info(f"Model loaded successfully: {self.model_path.name}")
            return model

        except Exception as e:
            error_msg = config.ERROR_MESSAGES_RU["model_load_failed"].format(
                model_name=self.model_path.name
            )
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"{error_msg}: {str(e)}")

    def detect_faces(
        self,
        image: np.ndarray,
        return_crops: bool = False,
        min_face_size: int = config.MIN_FACE_SIZE_PIXELS
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Note: Uses person detection from YOLO and estimates face location
        as upper 1/3 of person bounding box.

        Args:
            image: Input image as numpy array (BGR format)
            return_crops: If True, include cropped face images in results
            min_face_size: Minimum face size in pixels (width or height)

        Returns:
            List of FaceDetection objects

        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            logger.warning("Invalid image provided for detection")
            return []

        try:
            # Prepare inference parameters
            inference_params = {
                'conf': self.confidence_threshold,
                'iou': self.iou_threshold,
                'verbose': False,
                'classes': [0],  # Class 0 is 'person' in COCO dataset
                'device': self.device
            }

            # Add GPU optimizations
            if self.device == "cuda":
                if hasattr(config, 'YOLO_IMAGE_SIZE'):
                    inference_params['imgsz'] = config.YOLO_IMAGE_SIZE
                if hasattr(config, 'YOLO_USE_HALF_PRECISION') and config.YOLO_USE_HALF_PRECISION:
                    inference_params['half'] = True

            # Run inference
            results = self.model(image, **inference_params)

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # Get bounding box coordinates for person
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Estimate face location (upper 1/3 of person bbox)
                    person_height = py2 - py1
                    person_width = px2 - px1

                    # Face is typically in upper 1/3, centered horizontally
                    face_height = int(person_height * 0.35)  # Upper 35% of body
                    face_width = int(person_width * 0.8)     # 80% of body width

                    # Calculate face bounding box
                    x1 = px1 + (person_width - face_width) // 2
                    y1 = py1
                    x2 = x1 + face_width
                    y2 = py1 + face_height

                    # Ensure bounds are within image
                    img_h, img_w = image.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)

                    # Filter by minimum size
                    width = x2 - x1
                    height = y2 - y1

                    if width < min_face_size or height < min_face_size:
                        continue

                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = 0  # We use person class

                    # Create detection object
                    face_crop = None
                    if return_crops:
                        face_crop = self._extract_face_crop(image, (x1, y1, x2, y2))

                    detection = FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        face_crop=face_crop
                    )

                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} faces in image")
            return detections

        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            raise RuntimeError(f"Ошибка детекции лиц: {str(e)}")

    def _extract_face_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = config.FACE_CROP_PADDING
    ) -> np.ndarray:
        """
        Extract and return cropped face region from image.

        Args:
            image: Source image
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Pixels to add around face

        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop face
        face_crop = image[y1:y2, x1:x2].copy()

        return face_crop

    def detect_faces_batch(
        self,
        images: List[np.ndarray],
        return_crops: bool = False,
        min_face_size: int = config.MIN_FACE_SIZE_PIXELS
    ) -> List[List[FaceDetection]]:
        """
        Detect faces in multiple images (batch processing).

        Args:
            images: List of input images
            return_crops: If True, include cropped face images
            min_face_size: Minimum face size in pixels

        Returns:
            List of detection lists (one per image)

        Note:
            Batch processing is more efficient for multiple images
        """
        if not images:
            return []

        try:
            # Run batch inference
            results = self.model(
                images,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                classes=[0]  # Class 0 is 'person'
            )

            all_detections = []

            # Process each image's results
            for img_idx, result in enumerate(results):
                detections = []
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get person bbox
                        px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Estimate face location
                        person_height = py2 - py1
                        person_width = px2 - px1

                        face_height = int(person_height * 0.35)
                        face_width = int(person_width * 0.8)

                        x1 = px1 + (person_width - face_width) // 2
                        y1 = py1
                        x2 = x1 + face_width
                        y2 = py1 + face_height

                        # Ensure bounds
                        img_h, img_w = images[img_idx].shape[:2]
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_w, x2)
                        y2 = min(img_h, y2)

                        # Filter by minimum size
                        width = x2 - x1
                        height = y2 - y1

                        if width < min_face_size or height < min_face_size:
                            continue

                        confidence = float(box.conf[0])
                        class_id = 0

                        face_crop = None
                        if return_crops:
                            face_crop = self._extract_face_crop(
                                images[img_idx],
                                (x1, y1, x2, y2)
                            )

                        detection = FaceDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            face_crop=face_crop
                        )

                        detections.append(detection)

                all_detections.append(detections)

            logger.debug(f"Batch processed {len(images)} images, "
                        f"total detections: {sum(len(d) for d in all_detections)}")

            return all_detections

        except Exception as e:
            logger.error(f"Error during batch face detection: {str(e)}")
            raise RuntimeError(f"Ошибка пакетной детекции: {str(e)}")

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[FaceDetection],
        show_confidence: bool = True,
        color: Tuple[int, int, int] = config.BBOX_COLOR,
        thickness: int = config.BBOX_THICKNESS
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.

        Args:
            image: Input image
            detections: List of face detections
            show_confidence: Whether to show confidence scores
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Image with drawn detections
        """
        vis_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label with confidence
            if show_confidence:
                label = f"{detection.confidence:.2f}"
                label_size, _ = cv2.getTextSize(
                    label,
                    config.TEXT_FONT,
                    config.TEXT_SCALE,
                    config.TEXT_THICKNESS
                )

                # Draw background for text
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    config.TEXT_BG_COLOR,
                    -1
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 5),
                    config.TEXT_FONT,
                    config.TEXT_SCALE,
                    config.TEXT_COLOR,
                    config.TEXT_THICKNESS
                )

        return vis_image

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_path.name,
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_exists": self.model_path.exists()
        }


def filter_overlapping_detections(
    detections: List[FaceDetection],
    iou_threshold: float = 0.5
) -> List[FaceDetection]:
    """
    Remove overlapping detections using Non-Maximum Suppression.

    Args:
        detections: List of face detections
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence (descending)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep = []

    while sorted_detections:
        # Keep highest confidence detection
        current = sorted_detections.pop(0)
        keep.append(current)

        # Filter overlapping detections
        sorted_detections = [
            det for det in sorted_detections
            if calculate_iou(current.bbox, det.bbox) < iou_threshold
        ]

    return keep


def calculate_iou(bbox1: Tuple[int, int, int, int],
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union
