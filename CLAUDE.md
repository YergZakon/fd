# Emotion Recognition Application - Development Guide

## Project Overview

<background_information>
This is an advanced video emotion recognition application built with Streamlit. The application analyzes video files to detect faces and classify emotions using state-of-the-art computer vision and Vision-Language Models (VLM).

**Target User**: Legal professional in Kazakhstan creating digital legal services, with beginner programming level.

**Critical Requirement**: This is a PRODUCTION-GRADE application. DO NOT simplify functionality, use placeholders, or create MVP versions unless explicitly instructed. Every feature must be fully implemented and functional.

**Use Cases**:
- Court behavior analysis (stress/deception detection)
- Remote consultation emotional state assessment  
- Video evidence analysis and documentation
- Liveness detection for video identification
</background_information>

---

## Technology Stack

<technical_stack>
### Core Framework
- **Streamlit**: Main UI framework with session state management
- **Python 3.9+**: Base language

### Computer Vision & Detection
- **YOLOv11** (ultralytics): Face detection (latest YOLO version for best accuracy/speed)
- **ByteTrack/DeepSORT**: Multi-object tracking for consistent person IDs across frames
- **OpenCV (cv2)**: Video processing and frame manipulation

### Emotion Recognition (Multi-Model Approach)
- **Primary FER**: HSEmotion or trpakov/vit-face-expression from Hugging Face
- **VLM Context Analysis**: SmolVLM-Instruct or Florence-2 for contextual understanding
- **Fusion Layer**: Custom logic combining CNN-based FER + VLM analysis

### Optional Advanced Features
- **Silent-Face-Anti-Spoofing**: Liveness detection (minchul/cvpr24_fas_zero)
- **Micro-expression Analysis**: Custom implementation or specialized models

### Data Management
- **Pandas**: Results storage and manipulation
- **JSON/CSV**: Export formats
- **SQLite** (optional): Session persistence

### Visualization
- **Plotly**: Interactive charts and timelines
- **Matplotlib/Seaborn**: Statistical visualizations
</technical_stack>

---

## Architecture Principles

<architecture_guidelines>
### Pipeline Flow
```
Video Input → Frame Extraction → YOLOv11 Face Detection → Face Tracking (ID assignment) 
→ [Parallel Processing]:
  ├─ Traditional FER (CNN/ViT)
  └─ VLM Contextual Analysis
→ Emotion Fusion Layer → Manual Correction Interface → Results Export
```

### Modular Design
- **Separation of Concerns**: Each module handles ONE responsibility
- **Dependency Injection**: Models loaded once, passed to functions
- **Stateless Functions**: All processing functions should be pure where possible
- **Error Resilience**: Comprehensive try-catch with fallback strategies

### Performance Optimization
- **Progressive Loading**: Load models only when needed (lazy initialization)
- **Frame Sampling**: Configurable FPS reduction (process every N frames)
- **Batch Processing**: Where applicable, batch face crops for inference
- **Caching**: Use Streamlit @st.cache_resource for models, @st.cache_data for processed frames
- **Memory Management**: Clear large objects after processing, implement garbage collection
</architecture_guidelines>

---

## Development Instructions

<development_workflow>
### Phase 1: Project Structure Setup
Create the following directory structure:
```
emotion-detection-app/
├── app.py
├── requirements.txt
├── config.py
├── modules/
│   ├── __init__.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py
│   │   ├── face_tracker.py
│   │   └── liveness_checker.py (optional)
│   ├── emotion_recognition/
│   │   ├── __init__.py
│   │   ├── traditional_fer.py
│   │   ├── vlm_analyzer.py
│   │   └── emotion_fusion.py
│   ├── video_processing/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py
│   │   └── video_annotator.py
│   └── utils/
│       ├── __init__.py
│       ├── data_manager.py
│       └── visualization.py
├── models/ (for cached models)
├── temp/ (for temporary files)
└── outputs/ (for results)
```

### Phase 2: Core Module Implementation
Implement modules in this order to manage dependencies:

1. **config.py**: All configuration constants, model paths, emotion labels
2. **frame_extractor.py**: Video loading, frame extraction with timestamps
3. **yolo_detector.py**: Face detection using YOLOv11
4. **face_tracker.py**: Tracking faces across frames with unique IDs
5. **traditional_fer.py**: CNN/ViT-based emotion recognition
6. **vlm_analyzer.py**: VLM contextual emotion analysis
7. **emotion_fusion.py**: Logic to combine FER + VLM results
8. **data_manager.py**: Save/load results, export functionality
9. **visualization.py**: Charts, timelines, annotated video generation
10. **app.py**: Streamlit interface orchestrating all modules

### Phase 3: Streamlit Interface
Build a professional multi-page/multi-tab interface with:
- **Upload Tab**: Video upload, processing parameters configuration
- **Processing Tab**: Real-time progress, frame-by-frame preview
- **Results Tab**: Data table, filtering, manual correction
- **Visualization Tab**: Emotion timeline, statistics, distribution charts
- **Export Tab**: Download results (CSV/JSON), annotated video

### Phase 4: Testing & Refinement
- Test with short videos (5-10 sec) first
- Test with multiple faces
- Test with challenging lighting conditions
- Validate emotion accuracy manually
- Performance profiling and optimization
</development_workflow>

---

## Critical Implementation Rules

<mandatory_requirements>
### DO NOT:
- ❌ Use placeholder functions (e.g., `pass`, `# TODO`, `raise NotImplementedError`)
- ❌ Hardcode file paths without using config.py
- ❌ Skip error handling in ANY function
- ❌ Create simplified "MVP" versions of features
- ❌ Use print() statements instead of proper logging
- ❌ Leave commented-out code blocks
- ❌ Implement features without proper docstrings

### DO:
- ✅ Implement FULL functionality for every feature
- ✅ Add comprehensive error handling with specific error messages
- ✅ Use type hints for all function signatures
- ✅ Write docstrings for every function and class
- ✅ Implement proper logging (using logging module)
- ✅ Add progress bars for long-running operations
- ✅ Validate all user inputs
- ✅ Handle edge cases (empty frames, no faces detected, etc.)
- ✅ Use context managers for file operations
- ✅ Implement cleanup functions for temporary files

### Code Quality Standards
- **Type Annotations**: Use typing module for all functions
- **Docstrings**: Google-style docstrings with Args, Returns, Raises sections
- **Error Messages**: Must be in Russian for user-facing errors, English for logs
- **Constants**: All magic numbers must be in config.py
- **Naming**: Use descriptive variable names (no single letters except loop indices)
</mandatory_requirements>

---

## Model Integration Guidelines

<model_integration>
### YOLOv11 Face Detection
```python
# Expected implementation pattern (not actual code):
- Load model using ultralytics.YOLO('yolov11n.pt')
- Configure for face detection class
- Set confidence threshold (default 0.5)
- Return bounding boxes with confidence scores
- Handle cases with 0 faces or multiple faces
```

### Traditional FER Model
```python
# Expected implementation pattern:
- Load from Hugging Face (e.g., trpakov/vit-face-expression)
- Preprocess face crops (resize, normalize)
- Return emotion probabilities for 7 classes:
  ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
- Implement confidence thresholding
```

### VLM Contextual Analysis
```python
# Expected implementation pattern:
- Use SmolVLM-Instruct or Florence-2
- Construct prompt: "Analyze this person's facial expression..."
- Include context if available (location type, time)
- Parse VLM response to extract emotion classification
- Implement fallback if VLM unavailable or errors
```

### Emotion Fusion Strategy
Implement weighted voting or probability fusion:
- Traditional FER weight: 0.6 (more reliable for basic emotions)
- VLM weight: 0.4 (adds context, but may hallucinate)
- If confidence scores differ significantly, flag for manual review
- Allow user to adjust weights in UI
</model_integration>

---

## Streamlit UI Requirements

<ui_specifications>
### Layout Structure
Use st.tabs() for main sections, st.columns() for layouts, st.sidebar for global settings

### Session State Management
Store in st.session_state:
- uploaded_video_path
- extracted_frames
- detection_results
- emotion_results (with manual corrections)
- processing_config
- current_tab

### User Experience
- Show loading spinners with descriptive text (in Russian)
- Display progress bars with percentage and ETA
- Preview frames during processing (every Nth frame)
- Allow manual correction with dropdown selects
- Implement undo/redo for corrections
- Auto-save results to avoid data loss

### Russian Language
All UI text must be in Russian:
- Labels, buttons, headers
- Error messages for users
- Help tooltips and instructions
- Export filenames should use Russian transliteration
</ui_specifications>

---

## Data Management & Export

<data_specifications>
### Results Schema
Each detection should store:
```json
{
  "frame_number": int,
  "timestamp": float,
  "face_id": int,
  "bbox": [x1, y1, x2, y2],
  "emotion_traditional": str,
  "confidence_traditional": float,
  "emotion_vlm": str,
  "confidence_vlm": float,
  "emotion_final": str,
  "confidence_final": float,
  "manually_corrected": bool,
  "notes": str (optional)
}
```

### Export Formats
1. **CSV**: Flat structure for spreadsheet analysis
2. **JSON**: Full nested structure with metadata
3. **Annotated Video**: Original video with bounding boxes and emotion labels
4. **Report PDF** (optional): Summary statistics and visualizations

### Session Persistence
- Auto-save processing state every N seconds
- Allow resume from last saved state
- Store in temp/ directory with unique session ID
- Cleanup old sessions after 24 hours
</data_specifications>

---

## Performance Targets

<performance_requirements>
- **Frame Processing**: < 1 second per frame on CPU, < 0.2s on GPU
- **Video Upload**: Support up to 500MB files
- **Face Detection**: Process 30 FPS source at 5-10 FPS analysis rate
- **Memory Usage**: < 4GB RAM for 10-minute 1080p video
- **UI Responsiveness**: No freezes, all long operations in background threads
- **Model Loading**: < 30 seconds for all models on first run
</performance_requirements>

---

## Testing Strategy

<testing_approach>
### Test Cases to Implement
1. **Single person, clear lighting, basic emotions**
2. **Multiple people in frame simultaneously**
3. **Poor lighting conditions**
4. **Partial face occlusion (hand, object)**
5. **Profile views and head rotation**
6. **Very short video (< 5 seconds)**
7. **Long video (> 5 minutes)**
8. **No faces detected scenario**
9. **Rapid emotion changes**

### Validation Metrics
- Face detection recall/precision
- Emotion classification accuracy (manual validation sample)
- Processing time per frame
- False positive rate for liveness (if implemented)
- User correction rate (indicates model accuracy)
</testing_approach>

---

## Progressive Development Approach

<development_phases>
When implementing, follow this order but FULLY implement each component:

**Stage 1**: Core video processing pipeline
- Video upload and frame extraction
- YOLOv11 face detection
- Basic emotion recognition with one model
- Simple results table

**Stage 2**: Advanced recognition
- Face tracking with consistent IDs
- VLM integration
- Emotion fusion layer
- Manual correction interface

**Stage 3**: Visualization & Export
- Interactive timeline
- Statistical charts
- Multiple export formats
- Annotated video generation

**Stage 4**: Polish & Optimization
- Performance optimization
- Error handling refinement
- UI/UX improvements
- Documentation

**Stage 5** (Optional): Advanced features
- Liveness detection
- Micro-expression analysis
- Multi-language support
- Cloud deployment configuration
</development_phases>

---

## Context Engineering Best Practices for This Project

<context_management>
### Just-In-Time Loading
- Don't load all frames into memory at once
- Load models lazily when first needed
- Stream video frame-by-frame where possible
- Use file paths/identifiers instead of loading full videos

### Memory Management
- Clear processed frames after analysis
- Implement explicit garbage collection after heavy operations
- Use generators for frame iteration
- Store only essential data in session state

### Note-Taking for Long Videos
For videos requiring multiple processing sessions:
- Maintain processing_log.json tracking progress
- Store last processed frame number
- Save intermediate results incrementally
- Allow resuming from checkpoints

### Tool Organization
Keep functions focused and single-purpose:
- Each module handles ONE domain (detection, emotion, export)
- No overlapping functionality between modules
- Clear input/output contracts
- Comprehensive docstrings explaining when to use each function
</context_management>

---

## Success Criteria

<project_completion>
The application is considered complete when:

✅ User can upload video and see processing progress
✅ Faces are detected with > 90% recall on clear videos
✅ Emotions are classified and displayed with confidence scores
✅ Multiple people in frame are tracked with unique IDs
✅ Traditional FER and VLM both contribute to final emotion
✅ User can manually correct any emotion classification
✅ Results exportable in CSV, JSON, and annotated video
✅ Interactive visualizations show emotion timeline and statistics
✅ All error cases handled gracefully with Russian error messages
✅ Application runs without crashes on test videos
✅ Code is fully documented with docstrings
✅ No placeholder functions or TODO comments
✅ Performance meets targets specified above

**Definition of "Done"**: A legal professional with beginner coding skills can run the application, process their court videos, correct any misclassifications, and export professional reports - all without encountering errors or confusion.
</project_completion>

---

## Additional Notes

<important_reminders>
- **User Context**: The developer is a lawyer, not a professional programmer. Code should be well-commented and self-explanatory.
- **Language**: UI in Russian, code comments in English, docstrings in English.
- **Legal Domain**: Consider privacy implications - add warnings about sensitive video handling.
- **Deployment**: Prepare for local deployment (no cloud dependencies for core functionality).
- **Documentation**: Create a separate USER_GUIDE.md in Russian explaining how to use the application.
</important_reminders>

---

**Remember**: This is not an MVP. This is not a prototype. This is a production application. Every feature must work completely. No shortcuts, no placeholders, no "we'll add this later" comments.