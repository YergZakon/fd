"""
Emotion Fusion Module

Combines results from traditional FER and VLM models using weighted fusion.
"""

import logging
from typing import Dict, Tuple, Optional

import config


logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class EmotionFusion:
    """Fuses emotion predictions from multiple models."""

    def __init__(
        self,
        fer_weight: float = config.FER_WEIGHT,
        vlm_weight: float = config.VLM_WEIGHT
    ):
        """
        Initialize emotion fusion.

        Args:
            fer_weight: Weight for traditional FER model (0-1)
            vlm_weight: Weight for VLM model (0-1)
        """
        # Normalize weights
        total = fer_weight + vlm_weight
        self.fer_weight = fer_weight / total
        self.vlm_weight = vlm_weight / total

        logger.info(f"EmotionFusion initialized: FER={self.fer_weight:.2f}, VLM={self.vlm_weight:.2f}")

    def fuse_predictions(
        self,
        fer_probs: Dict[str, float],
        vlm_emotion: Optional[str] = None,
        vlm_confidence: float = 0.0
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Fuse FER and VLM predictions.

        Args:
            fer_probs: FER emotion probabilities
            vlm_emotion: VLM predicted emotion (optional)
            vlm_confidence: VLM confidence score

        Returns:
            Tuple of (final_emotion, confidence, fused_probabilities)
        """
        # Initialize fused probabilities with FER weighted scores
        fused_probs = {emotion: prob * self.fer_weight for emotion, prob in fer_probs.items()}

        # Add VLM contribution if available
        if vlm_emotion and vlm_emotion in config.EMOTION_LABELS:
            # Distribute VLM confidence across emotions
            for emotion in fused_probs:
                if emotion == vlm_emotion:
                    fused_probs[emotion] += vlm_confidence * self.vlm_weight
                else:
                    fused_probs[emotion] += (1 - vlm_confidence) * self.vlm_weight / (len(config.EMOTION_LABELS) - 1)

        # Get dominant emotion
        final_emotion = max(fused_probs.items(), key=lambda x: x[1])

        return final_emotion[0], final_emotion[1], fused_probs

    def needs_manual_review(
        self,
        fer_emotion: str,
        vlm_emotion: Optional[str],
        fer_confidence: float,
        vlm_confidence: float
    ) -> bool:
        """
        Determine if prediction needs manual review.

        Args:
            fer_emotion: FER predicted emotion
            vlm_emotion: VLM predicted emotion
            fer_confidence: FER confidence
            vlm_confidence: VLM confidence

        Returns:
            True if manual review recommended
        """
        # Different predictions with similar confidence
        if vlm_emotion and fer_emotion != vlm_emotion:
            conf_diff = abs(fer_confidence - vlm_confidence)
            if conf_diff < config.CONFIDENCE_DIFF_THRESHOLD:
                return True

        # Low confidence predictions
        if fer_confidence < config.MIN_CONFIDENCE_THRESHOLD:
            return True

        return False
