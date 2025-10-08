"""
Visualization Module

Creates charts, timelines, and visual analytics for emotion data.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

import config


logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class EmotionVisualizer:
    """Creates visualizations for emotion detection results."""

    def __init__(self):
        """Initialize visualizer."""
        logger.info("EmotionVisualizer initialized")

    def create_emotion_timeline(
        self,
        df: pd.DataFrame,
        face_id: Optional[int] = None
    ) -> go.Figure:
        """
        Create timeline of emotions over time.

        Args:
            df: DataFrame with results
            face_id: Filter by specific face ID

        Returns:
            Plotly figure
        """
        if df.empty:
            return go.Figure()

        # Filter by face_id if specified
        if face_id is not None:
            df = df[df['face_id'] == face_id]

        fig = px.scatter(
            df,
            x='timestamp',
            y='emotion_final',
            color='emotion_final',
            color_discrete_map=config.EMOTION_COLORS,
            title=f"Временная шкала эмоций" + (f" (Лицо #{face_id})" if face_id else ""),
            labels={'timestamp': 'Время (сек)', 'emotion_final': 'Эмоция'},
            hover_data=['confidence_final', 'frame_number']
        )

        fig.update_layout(
            height=400,
            xaxis_title="Время (секунды)",
            yaxis_title="Эмоция",
            showlegend=True
        )

        return fig

    def create_emotion_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart of emotion distribution.

        Args:
            df: DataFrame with results

        Returns:
            Plotly figure
        """
        if df.empty:
            return go.Figure()

        emotion_counts = df['emotion_final'].value_counts()

        # Map to Russian labels
        labels = [config.EMOTION_LABELS_RU.get(emotion, emotion) for emotion in emotion_counts.index]
        colors = [config.EMOTION_COLORS.get(emotion, '#808080') for emotion in emotion_counts.index]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=emotion_counts.values,
            marker=dict(colors=colors)
        )])

        fig.update_layout(
            title="Распределение эмоций",
            height=400
        )

        return fig

    def create_confidence_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create confidence score plot over time.

        Args:
            df: DataFrame with results

        Returns:
            Plotly figure
        """
        if df.empty:
            return go.Figure()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence_final'],
            mode='lines+markers',
            name='Уверенность',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title="Уверенность модели во времени",
            xaxis_title="Время (секунды)",
            yaxis_title="Уверенность (0-1)",
            height=300,
            yaxis=dict(range=[0, 1])
        )

        return fig

    def create_face_statistics(self, df: pd.DataFrame) -> go.Figure:
        """
        Create bar chart of detections per face.

        Args:
            df: DataFrame with results

        Returns:
            Plotly figure
        """
        if df.empty:
            return go.Figure()

        face_counts = df['face_id'].value_counts().sort_index()

        fig = go.Figure(data=[
            go.Bar(
                x=[f"Лицо #{fid}" for fid in face_counts.index],
                y=face_counts.values,
                marker_color='lightblue'
            )
        ])

        fig.update_layout(
            title="Количество обнаружений по лицам",
            xaxis_title="ID лица",
            yaxis_title="Количество кадров",
            height=300
        )

        return fig
