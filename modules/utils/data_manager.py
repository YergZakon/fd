"""
Data Management Module

Handles saving, loading, and exporting emotion detection results.
"""

import json
import csv
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import config


logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DataManager:
    """Manages emotion detection data persistence and export."""

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize data manager.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[Dict[str, Any]] = []

        logger.info(f"DataManager initialized for session: {self.session_id}")

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a detection result."""
        self.results.append(result)

    def add_results_batch(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple results at once."""
        self.results.extend(results)

    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as pandas DataFrame.

        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Convert timestamp to readable format
        if 'timestamp' in df.columns:
            df['timestamp_formatted'] = df['timestamp'].apply(
                lambda x: f"{int(x//60):02d}:{int(x%60):02d}.{int((x%1)*1000):03d}"
            )

        return df

    def export_csv(self, output_path: str | Path) -> None:
        """
        Export results to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_results_df()

        if df.empty:
            logger.warning("No results to export")
            return

        df.to_csv(
            output_path,
            index=False,
            encoding=config.CSV_ENCODING,
            sep=config.CSV_SEPARATOR
        )

        logger.info(f"Results exported to CSV: {output_path}")

    def export_json(self, output_path: str | Path) -> None:
        """
        Export results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "session_id": self.session_id,
            "export_time": datetime.now().isoformat(),
            "total_detections": len(self.results),
            "results": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                export_data,
                f,
                indent=config.JSON_INDENT,
                ensure_ascii=config.JSON_ENSURE_ASCII
            )

        logger.info(f"Results exported to JSON: {output_path}")

    def save_session(self, checkpoint_path: Optional[Path] = None) -> Path:
        """
        Save current session state.

        Args:
            checkpoint_path: Path to save checkpoint

        Returns:
            Path where session was saved
        """
        if checkpoint_path is None:
            checkpoint_path = config.TEMP_DIR / f"session_{self.session_id}.json"

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": self.session_id,
            "save_time": datetime.now().isoformat(),
            "results": self.results
        }

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Session saved: {checkpoint_path}")
        return checkpoint_path

    def load_session(self, checkpoint_path: Path) -> None:
        """
        Load session from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        self.session_id = session_data.get('session_id', self.session_id)
        self.results = session_data.get('results', [])

        logger.info(f"Session loaded: {len(self.results)} results")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected results.

        Returns:
            Dictionary with statistics
        """
        if not self.results:
            return {"total_detections": 0}

        df = self.get_results_df()

        stats = {
            "total_detections": len(df),
            "unique_faces": df['face_id'].nunique() if 'face_id' in df.columns else 0,
            "total_frames": df['frame_number'].nunique() if 'frame_number' in df.columns else 0,
        }

        # Emotion distribution
        if 'emotion_final' in df.columns:
            emotion_counts = df['emotion_final'].value_counts().to_dict()
            stats['emotion_distribution'] = emotion_counts

        return stats

    def clear_results(self) -> None:
        """Clear all results."""
        self.results.clear()
        logger.info("Results cleared")
