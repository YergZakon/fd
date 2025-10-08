"""
Traditional Facial Emotion Recognition Module

Implements CNN/ViT-based emotion recognition using Hugging Face models.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
    import torch
except ImportError:
    raise ImportError("transformers and torch required. Install with: pip install transformers torch")

import config


logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class TraditionalFER:
    """Traditional Facial Emotion Recognition using pretrained models."""

    def __init__(
        self,
        model_name: str = config.FER_MODEL_NAME,
        device: str = config.YOLO_DEVICE
    ):
        """
        Initialize FER model.

        Args:
            model_name: Hugging Face model identifier
            device: Device for inference ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

        self._load_model()
        logger.info(f"TraditionalFER initialized with {model_name}")

    def _load_model(self) -> None:
        """Load model and processor."""
        try:
            device_idx = 0 if self.device == "cuda" else -1
            self.model = pipeline(
                "image-classification",
                model=self.model_name,
                device=device_idx
            )
            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load FER model: {str(e)}")
            raise RuntimeError(f"Ошибка загрузки модели эмоций: {str(e)}")

    def predict_emotion(
        self,
        face_image: np.ndarray,
        top_k: int = 7
    ) -> Dict[str, float]:
        """
        Predict emotion from face image.

        Args:
            face_image: Face crop as numpy array (BGR)
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Invalid face image")
            return {emotion: 0.0 for emotion in config.EMOTION_LABELS}

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Run inference
            results = self.model(rgb_image, top_k=top_k)

            # Convert to standard format
            emotion_probs = {emotion: 0.0 for emotion in config.EMOTION_LABELS}

            for result in results:
                label = result['label'].lower()
                score = result['score']

                # Map model labels to standard emotions
                if label in emotion_probs:
                    emotion_probs[label] = score

            return emotion_probs

        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            return {emotion: 0.0 for emotion in config.EMOTION_LABELS}

    def predict_batch(
        self,
        face_images: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """
        Predict emotions for batch of faces.

        Args:
            face_images: List of face crops

        Returns:
            List of emotion probability dictionaries
        """
        if not face_images:
            return []

        try:
            # Convert all to RGB
            rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in face_images]

            # Batch inference
            batch_results = self.model(rgb_images, top_k=7, batch_size=config.BATCH_SIZE_FER)

            # Process results
            all_emotions = []
            for results in batch_results:
                emotion_probs = {emotion: 0.0 for emotion in config.EMOTION_LABELS}
                for result in results:
                    label = result['label'].lower()
                    if label in emotion_probs:
                        emotion_probs[label] = result['score']
                all_emotions.append(emotion_probs)

            return all_emotions

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return [{emotion: 0.0 for emotion in config.EMOTION_LABELS} for _ in face_images]

    def get_dominant_emotion(
        self,
        emotion_probs: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Get dominant emotion and its confidence.

        Args:
            emotion_probs: Dictionary of emotion probabilities

        Returns:
            Tuple of (emotion_label, confidence)
        """
        if not emotion_probs:
            return "neutral", 0.0

        dominant = max(emotion_probs.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
