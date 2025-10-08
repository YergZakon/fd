# GPU Setup для RTX 5090

## Текущая конфигурация

✅ **CUDA обнаружена**: PyTorch 2.7.1+cu128
✅ **GPU**: NVIDIA GeForce RTX 5090 Laptop GPU
✅ **CUDA версия**: 12.8

## Оптимизации применены

### 1. Конфигурация ([config.py](config.py))
```python
YOLO_MODEL_NAME = "yolo11l.pt"  # Large model (баланс скорость/точность)
YOLO_DEVICE = "cuda"             # GPU ускорение
YOLO_IMAGE_SIZE = 1280           # Высокое разрешение
YOLO_USE_HALF_PRECISION = True   # FP16 для 2x ускорения
YOLO_BATCH_SIZE = 8              # Оптимально для laptop GPU
```

### 2. YOLO Detector оптимизации
- ✅ Автоматическое использование GPU
- ✅ FP16 (half precision) inference
- ✅ Динамическое разрешение изображений
- ✅ Batch processing support

## Тестирование производительности

Запустите бенчмарк:
```bash
python test_gpu_performance.py
```

### Ожидаемые результаты для RTX 5090 Laptop:

| Модель | Разрешение | FPS (ожидаемо) | Использование |
|--------|-----------|----------------|---------------|
| yolo11l.pt | 640px | ~100-150 FPS | Real-time |
| yolo11l.pt | 1280px | ~40-60 FPS | Видео анализ ⭐ |
| yolo11l.pt | 1920px | ~15-25 FPS | Высокая точность |

## Быстрый старт

### 1. Проверка GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Запуск приложения
```bash
streamlit run app.py
```

### 3. Тест производительности
```bash
python test_gpu_performance.py
```

## Настройка производительности

### Максимальная скорость (для real-time)
В [config.py](config.py):
```python
YOLO_MODEL_NAME = "yolo11m.pt"  # Medium model
YOLO_IMAGE_SIZE = 640
DEFAULT_FPS_SAMPLE_RATE = 1  # Каждый кадр
```
Ожидаемо: **100+ FPS**

### Баланс (рекомендуется) ⭐
```python
YOLO_MODEL_NAME = "yolo11l.pt"  # Large model
YOLO_IMAGE_SIZE = 1280
DEFAULT_FPS_SAMPLE_RATE = 2  # Каждый 2-й кадр
```
Ожидаемо: **40-60 FPS**

### Максимальная точность
```python
YOLO_MODEL_NAME = "yolo11x.pt"  # Extra-large model
YOLO_IMAGE_SIZE = 1920
DEFAULT_FPS_SAMPLE_RATE = 5  # Каждый 5-й кадр
```
Ожидаемо: **15-25 FPS**

## Решение проблем

### Ошибка "CUDA out of memory"
```python
# В config.py уменьшите:
YOLO_BATCH_SIZE = 4  # Вместо 8
YOLO_IMAGE_SIZE = 640  # Вместо 1280
```

### Медленная обработка
1. Проверьте что используется GPU:
```python
import torch
print(torch.cuda.is_available())  # Должно быть True
```

2. Включите FP16:
```python
YOLO_USE_HALF_PRECISION = True
```

3. Используйте легче модель:
```python
YOLO_MODEL_NAME = "yolo11m.pt"
```

## Мониторинг GPU

### Во время работы
```bash
# Windows
nvidia-smi

# Или непрерывно
nvidia-smi -l 1
```

Ожидаемо:
- GPU Utilization: 70-95%
- Memory: 2-6 GB / 16 GB
- Temperature: 60-80°C

## Дополнительные оптимизации (опционально)

### TensorRT Export (дополнительное 2-3x ускорение)
```python
from ultralytics import YOLO

# Экспорт в TensorRT (один раз)
model = YOLO("yolo11l.pt")
model.export(format="engine", half=True, device=0)

# Использование
model_trt = YOLO("yolo11l.engine")
# Ожидаемо: 5-10 мс/кадр (100-200 FPS!)
```

## Сравнение моделей YOLO11

| Модель | Размер | mAP | Скорость (RTX 5090) | Память |
|--------|--------|-----|---------------------|--------|
| yolo11n.pt | 5 MB | 39.5 | ~200 FPS | 0.5 GB |
| yolo11s.pt | 18 MB | 47.0 | ~150 FPS | 1 GB |
| yolo11m.pt | 40 MB | 51.5 | ~100 FPS | 2 GB |
| yolo11l.pt | 50 MB | 53.4 | ~50 FPS | 3 GB |
| yolo11x.pt | 110 MB | 54.7 | ~25 FPS | 5 GB |

## Файлы конфигурации

- 📄 [config.py](config.py) - Основная конфигурация
- 📄 [test_gpu_performance.py](test_gpu_performance.py) - Бенчмарк скрипт
- 📄 [modules/detectors/yolo_detector.py](modules/detectors/yolo_detector.py) - YOLO детектор

---

**Статус**: ✅ GPU оптимизация активна
**Последнее обновление**: 2025-10-08
