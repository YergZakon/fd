# 🎭 Emotion Recognition Application

Профессиональная система анализа эмоций на видео для юридических целей.

## Описание

Приложение для автоматического распознавания эмоций на видео с использованием современных моделей компьютерного зрения и глубокого обучения. Разработано специально для юридических профессионалов в Казахстане.

### Основные возможности

- ✅ Детекция лиц с использованием YOLOv11
- ✅ Распознавание 7 базовых эмоций (злость, отвращение, страх, счастье, нейтральное, грусть, удивление)
- ✅ Отслеживание нескольких лиц в кадре с уникальными ID
- ✅ Интерактивная веб-интерфейс на Streamlit
- ✅ Экспорт результатов в CSV и JSON
- ✅ Интерактивные графики и визуализации
- ✅ Поддержка видео до 500 МБ

### Технологический стек

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, YOLOv11 (Ultralytics)
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Tracking**: IoU-based multi-object tracking
- **Visualization**: Plotly, Matplotlib
- **Data**: Pandas, NumPy

## Установка

### Требования

- Python 3.9 или выше
- pip
- (Опционально) CUDA для GPU ускорения

### Инструкция по установке

1. Клонируйте репозиторий:
```bash
git clone https://github.com/YergZakon/fd.git
cd fd
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Запуск приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`

### Рабочий процесс

1. **Загрузка видео** - загрузите видео файл через веб-интерфейс
2. **Обработка** - нажмите кнопку "Начать обработку" и дождитесь завершения
3. **Анализ результатов** - просмотрите статистику, графики и таблицу данных
4. **Экспорт** - скачайте результаты в формате CSV или JSON

### Поддерживаемые форматы видео

- MP4
- AVI
- MOV
- MKV
- WMV
- FLV

## Структура проекта

```
fd/
├── app.py                          # Главное приложение Streamlit
├── config.py                       # Конфигурация и константы
├── requirements.txt                # Python зависимости
├── modules/
│   ├── detectors/                  # Модули детекции и трекинга
│   │   ├── yolo_detector.py       # YOLOv11 детектор лиц
│   │   └── face_tracker.py        # Трекинг лиц
│   ├── emotion_recognition/        # Модули распознавания эмоций
│   │   ├── traditional_fer.py     # CNN/ViT эмоции
│   │   ├── vlm_analyzer.py        # VLM контекстный анализ
│   │   └── emotion_fusion.py      # Объединение результатов
│   ├── video_processing/           # Обработка видео
│   │   └── frame_extractor.py     # Извлечение кадров
│   └── utils/                      # Утилиты
│       ├── data_manager.py        # Управление данными
│       └── visualization.py       # Визуализация
├── models/                         # Кэш моделей (создается автоматически)
├── temp/                           # Временные файлы
└── outputs/                        # Экспортированные результаты
```

## Конфигурация

Основные настройки находятся в файле `config.py`:

- `YOLO_CONFIDENCE_THRESHOLD` - порог уверенности детекции (по умолчанию 0.5)
- `DEFAULT_FPS_SAMPLE_RATE` - обработка каждого N-го кадра (по умолчанию 5)
- `MAX_VIDEO_SIZE_MB` - максимальный размер видео (по умолчанию 500 МБ)
- `FER_WEIGHT` и `VLM_WEIGHT` - веса для объединения моделей

## Примеры использования

### Программный доступ к модулям

```python
from modules.video_processing.frame_extractor import VideoFrameExtractor
from modules.detectors.yolo_detector import YOLOFaceDetector

# Загрузка видео
extractor = VideoFrameExtractor("video.mp4")
metadata = extractor.get_metadata()

# Детекция лиц
detector = YOLOFaceDetector()
for frame_num, timestamp, frame in extractor.extract_frames(sample_rate=5):
    detections = detector.detect_faces(frame)
    print(f"Frame {frame_num}: {len(detections)} faces detected")
```

## Решение проблем

### Ошибка загрузки модели YOLO

Если модель не загружается автоматически, скачайте вручную:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
mv yolo11n.pt models/
```

### Медленная обработка видео

- Увеличьте `DEFAULT_FPS_SAMPLE_RATE` в config.py
- Используйте GPU (установите `YOLO_DEVICE = "cuda"`)
- Уменьшите разрешение видео перед обработкой

### Ошибки с памятью

- Обрабатывайте более короткие видео
- Увеличьте `DEFAULT_FPS_SAMPLE_RATE`
- Закройте другие приложения

## Разработка

### Запуск в режиме разработки

```bash
# Включить debug режим
export DEBUG_MODE=True  # Linux/Mac
set DEBUG_MODE=True     # Windows

streamlit run app.py
```

### Тестирование

```python
# Запуск с коротким тестовым видео
python -c "from modules.video_processing.frame_extractor import VideoFrameExtractor; print(VideoFrameExtractor('test.mp4').get_metadata())"
```

## Производительность

- **CPU**: ~0.5-1 сек на кадр
- **GPU (CUDA)**: ~0.1-0.2 сек на кадр
- **Рекомендуемое железо**: 8+ ГБ RAM, современный CPU или NVIDIA GPU

## Лицензия

Этот проект разработан для образовательных и профессиональных целей.

## Авторы

Разработано с использованием Claude Code от Anthropic.

## Поддержка

Для вопросов и предложений создайте Issue в GitHub: https://github.com/YergZakon/fd/issues

---

**Примечание**: Приложение предназначено для анализа видео в рамках законной деятельности. Соблюдайте законы о конфиденциальности и защите данных.
