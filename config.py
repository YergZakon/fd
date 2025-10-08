"""
Configuration file for Emotion Recognition Application

Contains all constants, model paths, and configuration parameters.
Centralized configuration management for easy adjustment.
"""

import os
from pathlib import Path
from typing import List, Dict

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ==================== MODEL CONFIGURATION ====================

# YOLOv11 Face Detection
YOLO_MODEL_NAME = "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = "cpu"  # Options: "cpu", "cuda", "mps" (for Apple Silicon)

# Traditional FER Model
FER_MODEL_NAME = "trpakov/vit-face-expression"  # Hugging Face model ID
FER_MODEL_CACHE_DIR = MODELS_DIR / "fer_cache"
FER_CONFIDENCE_THRESHOLD = 0.4

# VLM Model Configuration
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"  # Options: SmolVLM, Florence-2
VLM_MODEL_CACHE_DIR = MODELS_DIR / "vlm_cache"
VLM_MAX_NEW_TOKENS = 100
VLM_TEMPERATURE = 0.3
VLM_ENABLED = True  # Set to False to disable VLM analysis

# ==================== EMOTION LABELS ====================

# Standard 7 emotion classes
EMOTION_LABELS: List[str] = [
    "anger",      # злость
    "disgust",    # отвращение
    "fear",       # страх
    "happiness",  # счастье
    "neutral",    # нейтральное
    "sadness",    # грусть
    "surprise"    # удивление
]

# Russian translations for UI
EMOTION_LABELS_RU: Dict[str, str] = {
    "anger": "Злость",
    "disgust": "Отвращение",
    "fear": "Страх",
    "happiness": "Счастье",
    "neutral": "Нейтральное",
    "sadness": "Грусть",
    "surprise": "Удивление"
}

# Color mapping for visualization
EMOTION_COLORS: Dict[str, str] = {
    "anger": "#FF0000",      # Red
    "disgust": "#9400D3",    # Dark Violet
    "fear": "#8B008B",       # Dark Magenta
    "happiness": "#FFD700",  # Gold
    "neutral": "#808080",    # Gray
    "sadness": "#0000FF",    # Blue
    "surprise": "#FF1493"    # Deep Pink
}

# ==================== EMOTION FUSION PARAMETERS ====================

# Weights for combining FER and VLM results
FER_WEIGHT = 0.6  # Traditional FER model weight
VLM_WEIGHT = 0.4  # VLM contextual analysis weight

# Confidence difference threshold for manual review flagging
CONFIDENCE_DIFF_THRESHOLD = 0.3

# Minimum confidence for accepting prediction
MIN_CONFIDENCE_THRESHOLD = 0.3

# ==================== VIDEO PROCESSING ====================

# Frame extraction settings
DEFAULT_FPS_SAMPLE_RATE = 5  # Process every Nth frame
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

# Frame preprocessing
FRAME_RESIZE_WIDTH = 1280  # Resize frames to this width (maintains aspect ratio)
FACE_CROP_PADDING = 20  # Pixels to add around detected face

# ==================== TRACKING CONFIGURATION ====================

# Face tracking parameters
TRACKER_TYPE = "bytetrack"  # Options: "bytetrack", "deepsort"
TRACKER_MAX_AGE = 30  # Frames to keep track without detection
TRACKER_MIN_HITS = 3  # Minimum detections before track is confirmed
TRACKER_IOU_THRESHOLD = 0.3

# ==================== SESSION MANAGEMENT ====================

# Auto-save interval (seconds)
AUTO_SAVE_INTERVAL = 60

# Session cleanup (hours)
SESSION_CLEANUP_HOURS = 24

# Maximum session state size (MB)
MAX_SESSION_STATE_SIZE_MB = 100

# ==================== PERFORMANCE SETTINGS ====================

# Processing
BATCH_SIZE_FER = 8  # Number of faces to process in one batch
BATCH_SIZE_VLM = 1  # VLM typically processes one at a time

# Memory management
ENABLE_GARBAGE_COLLECTION = True
GC_INTERVAL_FRAMES = 50  # Run garbage collection every N frames

# Progress display
PROGRESS_UPDATE_INTERVAL = 5  # Update progress every N frames
FRAME_PREVIEW_INTERVAL = 10  # Show preview every N frames

# ==================== EXPORT SETTINGS ====================

# CSV export settings
CSV_ENCODING = "utf-8-sig"  # BOM for Excel compatibility
CSV_SEPARATOR = ","

# JSON export settings
JSON_INDENT = 2
JSON_ENSURE_ASCII = False  # Allow Cyrillic characters

# Annotated video settings
ANNOTATED_VIDEO_CODEC = "mp4v"  # Options: "mp4v", "avc1", "h264"
ANNOTATED_VIDEO_FPS = 30
BBOX_THICKNESS = 2
BBOX_COLOR = (0, 255, 0)  # Green in BGR
TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White in BGR
TEXT_BG_COLOR = (0, 0, 0)  # Black in BGR

# ==================== LOGGING CONFIGURATION ====================

# Logging level
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log file settings
LOG_FILE_PATH = PROJECT_ROOT / "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 3

# ==================== UI CONFIGURATION ====================

# Streamlit page config
PAGE_TITLE = "Анализ Эмоций на Видео"
PAGE_ICON = "🎭"
LAYOUT = "wide"

# Sidebar configuration
SHOW_ADVANCED_SETTINGS = True

# Tab names (in Russian)
TAB_UPLOAD = "📤 Загрузка"
TAB_PROCESSING = "⚙️ Обработка"
TAB_RESULTS = "📊 Результаты"
TAB_VISUALIZATION = "📈 Визуализация"
TAB_EXPORT = "💾 Экспорт"

# ==================== VLM PROMPT TEMPLATES ====================

VLM_EMOTION_PROMPT = """Проанализируйте выражение лица этого человека.
Определите одну доминирующую эмоцию из списка: злость, отвращение, страх, счастье, нейтральное, грусть, удивление.
Ответьте только названием эмоции на английском языке одним словом из списка: anger, disgust, fear, happiness, neutral, sadness, surprise."""

VLM_CONTEXT_PROMPT_TEMPLATE = """Проанализируйте выражение лица этого человека в контексте: {context}.
Определите одну доминирующую эмоцию из списка: злость, отвращение, страх, счастье, нейтральное, грусть, удивление.
Ответьте только названием эмоции на английском языке одним словом из списка: anger, disgust, fear, happiness, neutral, sadness, surprise."""

# ==================== ERROR MESSAGES (Russian) ====================

ERROR_MESSAGES_RU = {
    "video_too_large": f"Видео слишком большое. Максимальный размер: {MAX_VIDEO_SIZE_MB} МБ",
    "video_format_unsupported": f"Неподдерживаемый формат видео. Поддерживаемые форматы: {', '.join(SUPPORTED_VIDEO_FORMATS)}",
    "video_load_failed": "Не удалось загрузить видео. Проверьте файл.",
    "no_faces_detected": "Лица не обнаружены в видео",
    "model_load_failed": "Ошибка загрузки модели: {model_name}",
    "processing_failed": "Ошибка при обработке: {error}",
    "export_failed": "Ошибка при экспорте: {error}",
    "invalid_session": "Недействительная сессия. Начните заново.",
    "memory_limit": "Превышен лимит памяти. Попробуйте обработать более короткое видео.",
}

# ==================== SUCCESS MESSAGES (Russian) ====================

SUCCESS_MESSAGES_RU = {
    "video_uploaded": "Видео успешно загружено!",
    "processing_complete": "Обработка завершена!",
    "export_complete": "Экспорт завершен успешно!",
    "model_loaded": "Модель загружена: {model_name}",
}

# ==================== VALIDATION RULES ====================

# Video validation
MIN_VIDEO_DURATION_SEC = 0.5  # Minimum 0.5 seconds
MAX_VIDEO_DURATION_SEC = 600  # Maximum 10 minutes (adjustable)

# Face detection validation
MIN_FACE_SIZE_PIXELS = 30  # Minimum face bounding box size
MAX_FACES_PER_FRAME = 20  # Maximum faces to process per frame

# ==================== DEVELOPMENT/DEBUG SETTINGS ====================

DEBUG_MODE = False
SHOW_PERFORMANCE_METRICS = True
SAVE_DEBUG_FRAMES = False  # Save frames with detections for debugging
DEBUG_FRAMES_DIR = TEMP_DIR / "debug_frames"

if SAVE_DEBUG_FRAMES:
    DEBUG_FRAMES_DIR.mkdir(exist_ok=True)
