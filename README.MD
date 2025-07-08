# Player Tracking and Re-Identification in Sports Video

## About This Project

This project is a robust computer vision solution for a sports analytics challenge. It uses a YOLO detector and a Kalman Filter-based tracker to identify and maintain consistent IDs for each player, successfully handling occlusions and re-entry.

## 1. Project Overview

This project implements a real-time player tracking system for sports analytics, as per Option 2 of the assignment. The primary objective is to detect and assign a consistent ID to each player in a video feed. The system is designed to be robust against common tracking challenges like player occlusion and movement.

The core of the solution is a custom tracker that uses a **Kalman Filter** for motion prediction to ensure smooth and reliable tracking from frame to frame.

## 2. Features

- **Player Detection:** Utilizes a pre-trained YOLO model to detect players in each frame
- **Motion Prediction:** Employs a Kalman Filter for each track to predict the player's position in the subsequent frame, making the tracking robust to brief obstructions
- **Stable ID Assignment:** Uses the Hungarian algorithm to optimally match predicted positions with new detections based on their proximity (Intersection over Union)
- **Track Lifecycle Management:** A track is initialized in a "tentative" state (yellow box) and is promoted to "confirmed" (green box) only after being consistently tracked for several frames, preventing ghost tracks from false detections

## 3. Setup and Installation

Follow these steps to set up the environment and run the project.

### Prerequisites

- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/TashonBraganca/Player-Tracker-YOLOv11.git
cd Player-Tracker-YOLOv11
```

2. **Download the Model File:**
   
   ⚠️ **IMPORTANT:** The YOLO model file (`best.pt`) is too large for GitHub. Please download it from the original source.
   
   - **Download Link:** https://drive.google.com/file/d/1-5fOSHOSB9UXYP_enOoZNAMScrePVcMD/view
   - After downloading, place the `best.pt` file inside the `data` folder in the project directory

3. **Create and activate a virtual environment:**

   **For Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **For macOS/Linux:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install the required dependencies from `requirements.txt`:**

```bash
pip install -r requirements.txt
```

## 4. Usage

1. Place your input video file in the `data` folder as `15sec_input_720p.mp4`
2. Run the tracking script:

```bash
python main.py
```

3. The output video will be saved as `final_output_video.mp4` in the project root directory

## 5. File Structure

```
Player-Tracker-YOLOv11/
├── data/
│   ├── best.pt              # YOLO model (download required)
│   └── 15sec_input_720p.mp4 # Input video
├── main.py                  # Main tracking script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 6. Dependencies

The project requires the following Python packages:

- `ultralytics` - YOLO model inference
- `opencv-python` - Computer vision operations
- `numpy` - Numerical computations
- `scipy` - Scientific computing (Hungarian algorithm)
- `filterpy` - Kalman Filter implementation

## 7. How It Works

1. **Detection:** YOLO model detects players in each frame
2. **Prediction:** Kalman Filter predicts each tracked player's next position
3. **Association:** Hungarian algorithm matches predictions with new detections
4. **Update:** Successful matches update the track, unmatched detections create new tracks
5. **Lifecycle:** Tracks are managed with tentative/confirmed states to ensure stability

## 8. Output

The system outputs a video with:
- **Green boxes:** Confirmed player tracks with consistent IDs
- **Yellow boxes:** Tentative tracks (new detections being validated)
- **ID labels:** Each player maintains a unique ID throughout the video

## 9. Technical Details

- **Kalman Filter State:** [center_x, center_y, aspect_ratio, height, velocity_x, velocity_y, velocity_height]
- **Feature Extraction:** Color histogram features for appearance-based matching
- **IoU Threshold:** 0.3 for association decisions
- **Track Confirmation:** Requires 3 consecutive hits to confirm a track

## 10. Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 11. License

This project is licensed under the MIT License - see the LICENSE file for details.

## 12. Acknowledgments

- YOLO model for player detection
- Kalman Filter implementation from FilterPy
- Hungarian algorithm from SciPy
