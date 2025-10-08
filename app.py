"""
Emotion Recognition Application - Main Streamlit Interface

Production-grade video emotion detection application for legal professionals.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
import torch

# Fix for PyTorch 2.6+ weights_only issue with YOLO models
if hasattr(torch, 'serialization'):
    try:
        from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
        from ultralytics.engine.model import Model

        safe_classes = [DetectionModel, SegmentationModel, ClassificationModel, Model]
        torch.serialization.add_safe_globals(safe_classes)
    except Exception as e:
        print(f"Warning: Could not add Ultralytics safe globals: {e}")

import config
from modules.video_processing.frame_extractor import VideoFrameExtractor
from modules.detectors.yolo_detector import YOLOFaceDetector
from modules.detectors.face_tracker import SimpleFaceTracker
from modules.emotion_recognition.traditional_fer import TraditionalFER
from modules.emotion_recognition.vlm_analyzer import VLMEmotionAnalyzer
from modules.emotion_recognition.emotion_fusion import EmotionFusion
from modules.utils.data_manager import DataManager
from modules.utils.visualization import EmotionVisualizer


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()

    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


@st.cache_resource
def load_models():
    """Load all models (cached)."""
    try:
        with st.spinner("Загрузка моделей..."):
            detector = YOLOFaceDetector()
            fer_model = TraditionalFER()
            vlm_analyzer = VLMEmotionAnalyzer()
            emotion_fusion = EmotionFusion()

            logger.info("All models loaded successfully")
            return detector, fer_model, vlm_analyzer, emotion_fusion
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {str(e)}")
        logger.error(f"Model loading failed: {str(e)}")
        return None, None, None, None


def process_video(video_path: Path, detector, fer_model, vlm_analyzer, emotion_fusion):
    """
    Process video file and detect emotions.

    Args:
        video_path: Path to video file
        detector: Face detector
        fer_model: FER model
        vlm_analyzer: VLM analyzer
        emotion_fusion: Emotion fusion module
    """
    try:
        # Initialize components
        extractor = VideoFrameExtractor(video_path)
        tracker = SimpleFaceTracker()
        data_manager = st.session_state.data_manager

        metadata = extractor.get_metadata()

        st.info(f"Видео: {metadata.width}x{metadata.height}, "
                f"{metadata.fps:.1f} FPS, {metadata.duration:.1f} сек")

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_frames = metadata.total_frames
        processed_frames = 0

        # Process frames
        for frame_num, timestamp, frame in extractor.extract_frames(
            sample_rate=config.DEFAULT_FPS_SAMPLE_RATE
        ):
            # Detect faces
            detections = detector.detect_faces(frame, return_crops=True)

            # Track faces
            tracked_faces = tracker.update(detections)

            # Process each tracked face
            for tracked_face in tracked_faces:
                detection = tracked_face.detection

                if detection.face_crop is not None:
                    # FER prediction
                    fer_probs = fer_model.predict_emotion(detection.face_crop)
                    fer_emotion, fer_conf = fer_model.get_dominant_emotion(fer_probs)

                    # VLM prediction (if enabled)
                    vlm_emotion, vlm_conf = vlm_analyzer.analyze_emotion(detection.face_crop)

                    # Fuse predictions
                    final_emotion, final_conf, fused_probs = emotion_fusion.fuse_predictions(
                        fer_probs,
                        vlm_emotion,
                        vlm_conf
                    )

                    # Store result
                    result = {
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "face_id": tracked_face.track_id,
                        "bbox": detection.bbox,
                        "emotion_traditional": fer_emotion,
                        "confidence_traditional": fer_conf,
                        "emotion_vlm": vlm_emotion,
                        "confidence_vlm": vlm_conf,
                        "emotion_final": final_emotion,
                        "confidence_final": final_conf,
                        "manually_corrected": False,
                        "notes": ""
                    }

                    data_manager.add_result(result)

            # Update progress
            processed_frames += 1
            progress = min(frame_num / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Обработано кадров: {frame_num}/{total_frames}")

        progress_bar.progress(1.0)
        status_text.text("Обработка завершена!")

        st.session_state.processing_complete = True
        st.success(f"Обработано {len(data_manager.results)} детекций")

    except Exception as e:
        st.error(f"Ошибка при обработке видео: {str(e)}")
        logger.error(f"Video processing failed: {str(e)}")


def main():
    """Main application entry point."""
    initialize_session_state()

    st.title("🎭 Анализ Эмоций на Видео")

    st.markdown("""
    ### Профессиональная система распознавания эмоций для юридических целей

    **Возможности:**
    - Детекция лиц с использованием YOLOv11
    - Распознавание 7 базовых эмоций
    - Отслеживание нескольких лиц в кадре
    - Экспорт результатов в CSV/JSON
    - Интерактивные визуализации
    """)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        config.TAB_UPLOAD,
        config.TAB_PROCESSING,
        config.TAB_RESULTS,
        config.TAB_EXPORT
    ])

    # Tab 1: Upload
    with tab1:
        st.header("Загрузка видео")

        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Максимальный размер: 500 МБ"
        )

        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = Path(tmp_file.name)

            st.success("Видео загружено успешно!")

            # Show video preview
            st.video(str(st.session_state.video_path))

    # Tab 2: Processing
    with tab2:
        st.header("Обработка видео")

        if st.session_state.video_path is None:
            st.warning("Сначала загрузите видео во вкладке 'Загрузка'")
        else:
            if st.button("▶ Начать обработку", type="primary"):
                # Load models
                detector, fer_model, vlm_analyzer, emotion_fusion = load_models()

                if detector and fer_model and emotion_fusion:
                    st.session_state.models_loaded = True
                    process_video(
                        st.session_state.video_path,
                        detector,
                        fer_model,
                        vlm_analyzer,
                        emotion_fusion
                    )

    # Tab 3: Results
    with tab3:
        st.header("Результаты анализа")

        if not st.session_state.processing_complete:
            st.info("Результаты появятся после обработки видео")
        else:
            data_manager = st.session_state.data_manager
            df = data_manager.get_results_df()

            if not df.empty:
                # Statistics
                st.subheader("Статистика")
                stats = data_manager.get_statistics()

                col1, col2, col3 = st.columns(3)
                col1.metric("Всего детекций", stats['total_detections'])
                col2.metric("Уникальных лиц", stats.get('unique_faces', 0))
                col3.metric("Обработано кадров", stats.get('total_frames', 0))

                # Visualizations
                st.subheader("Визуализация")
                visualizer = EmotionVisualizer()

                # Emotion distribution
                st.plotly_chart(
                    visualizer.create_emotion_distribution(df),
                    use_container_width=True
                )

                # Timeline
                st.plotly_chart(
                    visualizer.create_emotion_timeline(df),
                    use_container_width=True
                )

                # Data table
                st.subheader("Таблица данных")
                st.dataframe(df, use_container_width=True)

    # Tab 4: Export
    with tab4:
        st.header("Экспорт результатов")

        if not st.session_state.processing_complete:
            st.info("Экспорт доступен после обработки видео")
        else:
            data_manager = st.session_state.data_manager

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📥 Скачать CSV"):
                    output_path = config.OUTPUTS_DIR / f"results_{data_manager.session_id}.csv"
                    data_manager.export_csv(output_path)

                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇ Загрузить CSV файл",
                            data=f,
                            file_name=output_path.name,
                            mime='text/csv'
                        )

            with col2:
                if st.button("📥 Скачать JSON"):
                    output_path = config.OUTPUTS_DIR / f"results_{data_manager.session_id}.json"
                    data_manager.export_json(output_path)

                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇ Загрузить JSON файл",
                            data=f,
                            file_name=output_path.name,
                            mime='application/json'
                        )

    # Sidebar
    with st.sidebar:
        st.header("Настройки")

        if config.SHOW_ADVANCED_SETTINGS:
            st.subheader("Параметры обработки")

            st.slider(
                "Частота кадров",
                min_value=1,
                max_value=30,
                value=config.DEFAULT_FPS_SAMPLE_RATE,
                help="Обрабатывать каждый N-ый кадр"
            )

            st.slider(
                "Порог уверенности",
                min_value=0.0,
                max_value=1.0,
                value=config.YOLO_CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Минимальная уверенность для детекции лиц"
            )

        st.divider()
        st.caption(f"Версия: {config.__version__ if hasattr(config, '__version__') else '1.0.0'}")


if __name__ == "__main__":
    main()
