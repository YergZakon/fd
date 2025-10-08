"""
GPU Performance Benchmark Script for RTX 5090

Tests YOLO11 face detection performance on GPU with various settings.
"""

import torch
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from modules.detectors.yolo_detector import YOLOFaceDetector


def print_gpu_info():
    """Print GPU information."""
    print("=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"✓ CUDA доступна: True")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA версия: {torch.version.cuda}")
        print(f"✓ PyTorch версия: {torch.__version__}")
        print(f"✓ Количество GPU: {torch.cuda.device_count()}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ Общая память GPU: {total_memory:.2f} GB")
    else:
        print("✗ CUDA недоступна!")
        sys.exit(1)

    print()


def benchmark_yolo(image_size: int = 1280, iterations: int = 100):
    """
    Benchmark YOLO detection speed.

    Args:
        image_size: Input image size
        iterations: Number of iterations for benchmark
    """
    print("=" * 70)
    print(f"BENCHMARK: YOLO11 @ {image_size}px")
    print("=" * 70)

    # Create detector
    print(f"Загрузка модели {config.YOLO_MODEL_NAME}...")
    detector = YOLOFaceDetector(
        confidence_threshold=0.5,
        device="cuda"
    )
    print(f"✓ Модель загружена на GPU\n")

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Warm-up
    print("Прогрев GPU (10 итераций)...")
    for _ in range(10):
        _ = detector.detect_faces(dummy_image)
    torch.cuda.synchronize()
    print("✓ Прогрев завершен\n")

    # Benchmark
    print(f"Запуск бенчмарка ({iterations} итераций)...")
    times = []

    for i in range(iterations):
        start = time.time()
        detections = detector.detect_faces(dummy_image)
        torch.cuda.synchronize()
        end = time.time()

        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Прогресс: {i + 1}/{iterations}")

    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time

    # GPU memory usage
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3

    print()
    print("РЕЗУЛЬТАТЫ:")
    print("-" * 70)
    print(f"Среднее время:      {avg_time:.2f} ± {std_time:.2f} мс/кадр")
    print(f"Минимум:            {min_time:.2f} мс")
    print(f"Максимум:           {max_time:.2f} мс")
    print(f"FPS:                {fps:.1f}")
    print(f"Память GPU (занято):{memory_allocated:.2f} GB")
    print(f"Память GPU (резерв):{memory_reserved:.2f} GB")
    print()

    # Performance evaluation
    if fps >= 50:
        print("✓ ОТЛИЧНО! Производительность превосходная для real-time обработки")
    elif fps >= 30:
        print("✓ ХОРОШО! Подходит для видео в реальном времени")
    elif fps >= 15:
        print("~ ПРИЕМЛЕМО! Подходит для анализа записанных видео")
    else:
        print("⚠ МЕДЛЕННО! Рассмотрите уменьшение разрешения или модели")

    print()
    return avg_time, fps


def test_different_resolutions():
    """Test performance at different resolutions."""
    print("=" * 70)
    print("ТЕСТ РАЗЛИЧНЫХ РАЗРЕШЕНИЙ")
    print("=" * 70)
    print()

    resolutions = [640, 1280, 1920]
    results = []

    for res in resolutions:
        avg_time, fps = benchmark_yolo(image_size=res, iterations=50)
        results.append((res, avg_time, fps))
        print()

    # Summary table
    print("=" * 70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(f"{'Разрешение':<15} {'Время (мс)':<15} {'FPS':<15} {'Рекомендация':<20}")
    print("-" * 70)

    for res, avg_time, fps in results:
        recommendation = ""
        if fps >= 50:
            recommendation = "Real-time ⭐"
        elif fps >= 30:
            recommendation = "Видео 30 FPS"
        elif fps >= 15:
            recommendation = "Анализ записей"
        else:
            recommendation = "Требуется оптимизация"

        print(f"{res}x{res:<10} {avg_time:<15.2f} {fps:<15.1f} {recommendation:<20}")

    print()


def test_batch_processing():
    """Test batch processing performance."""
    print("=" * 70)
    print("ТЕСТ BATCH PROCESSING")
    print("=" * 70)
    print()

    detector = YOLOFaceDetector(device="cuda")
    image_size = config.YOLO_IMAGE_SIZE if hasattr(config, 'YOLO_IMAGE_SIZE') else 1280
    batch_sizes = [1, 4, 8, 16]

    print(f"Разрешение изображения: {image_size}x{image_size}")
    print()

    for batch_size in batch_sizes:
        print(f"Тест batch_size = {batch_size}...")

        # Create batch of images
        images = [
            np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        # Warm-up
        for _ in range(5):
            _ = detector.detect_faces_batch(images)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(30):
            start = time.time()
            _ = detector.detect_faces_batch(images)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        time_per_image = avg_time / batch_size
        fps = 1000 / time_per_image

        print(f"  Batch time: {avg_time:.2f} мс")
        print(f"  Per image:  {time_per_image:.2f} мс")
        print(f"  FPS:        {fps:.1f}")
        print()


def main():
    """Main benchmark function."""
    print()
    print_gpu_info()

    # Check config
    print("=" * 70)
    print("ТЕКУЩАЯ КОНФИГУРАЦИЯ")
    print("=" * 70)
    print(f"Модель:            {config.YOLO_MODEL_NAME}")
    print(f"Device:            {config.YOLO_DEVICE}")
    print(f"Image size:        {getattr(config, 'YOLO_IMAGE_SIZE', 640)}")
    print(f"Half precision:    {getattr(config, 'YOLO_USE_HALF_PRECISION', False)}")
    print(f"Batch size:        {getattr(config, 'YOLO_BATCH_SIZE', 1)}")
    print()

    # Run tests
    try:
        # Basic benchmark
        benchmark_yolo(
            image_size=getattr(config, 'YOLO_IMAGE_SIZE', 1280),
            iterations=100
        )

        # Resolution comparison
        choice = input("Запустить тест различных разрешений? (y/n): ")
        if choice.lower() == 'y':
            test_different_resolutions()

        # Batch processing
        choice = input("Запустить тест batch processing? (y/n): ")
        if choice.lower() == 'y':
            test_batch_processing()

        print("=" * 70)
        print("✓ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("=" * 70)
        print()

    except KeyboardInterrupt:
        print("\n\nТест прерван пользователем")
    except Exception as e:
        print(f"\n\n✗ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
