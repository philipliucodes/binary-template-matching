# Binary Template Matching & Mouse Tracking

This project provides Python-based tools for **binary template matching** and **mouse tracking** using screen recordings. It utilizes **binarization** to convert input and template images into binary forms, allowing direct pixel-based matching. This technique is robust against variations in shading, contrast, and lighting conditions, making it ideal for detecting objects with distinct contours—such as cursors, UI elements, or symbols.

## Features

- Efficient binary template matching using alpha-aware binarization
- Batch processing of image(s) or video frames
- Mouse tracking via template matching with optional manual correction
- CSV export of results with timestamped detections

---

## Installation

Clone the repository and install dependencies using pip:

```bash
git clone https://github.com/yourusername/binary-template-matching.git
cd binary-template-matching
pip install -r requirements.txt
```

---

## Usage

### 1. Extracting Frames from a Video

The `frame_extractor.py` script extracts a specific frame from a video at a given timestamp.

#### Example:

```bash
python src/frame_extractor.py sample_video.mp4 00:44.250 --output extracted_frames
```

---

### 2. Template Matching with Binarization

The `template_matcher.py` script performs template matching between binarized images.

#### Example (single image vs single template):

```bash
python src/template_matcher.py input.jpg template.jpg --confidence_threshold 0.90 --white_threshold 200 --output results/
```

---

### 3. Mouse Tracking in Video with Optional Manual Review

The `mouse_tracker.py` script performs template matching directly on video frames and optionally queues uncertain results for manual review. It supports frame skipping via time intervals, focused searching based on previous frame matches, and output to a CSV file.

#### Example:

```bash
python src/mouse_tracker.py path/to/video.mp4 path/to/templates/ --interval 5 --output output_dir
```

**Important**: The template images in the directory should be sorted **alphabetically from most common to least common**. This means the most frequently occurring image (e.g., cursor type) should have a name that appears first alphabetically. This ensures optimal performance during template matching.

**Optional flags**:

- `--interval`: Time in seconds between frames to process (default: 5)
- `--output`: Output directory for CSV results (default: `output`)
- `--search_width` and `--search_height`: Dimensions of region to search around last match (default: 100x100)
- `--batch_size`: Number of frames per batch (default: 300)
- `--start_time` and `--end_time`: Optional start/end time in seconds to analyze a portion of the video

When the script can't find a match above the confidence threshold, it will open a window for **manual input**, where you can:

- Click to mark the object
- Press `a` to indicate "object not present"
- Press `s` to reuse the last manually selected coordinates
