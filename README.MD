Player Tracking and Re-Identification in Sports Video
About This Project
This project is a robust computer vision solution for a sports analytics challenge. It uses a YOLO detector and a Kalman Filter-based tracker to identify and maintain consistent IDs for each player, successfully handling occlusions and re-entry.

1. Project Overview
This project implements a real-time player tracking system for sports analytics, as per Option 2 of the assignment. The primary objective is to detect and assign a consistent ID to each player in a video feed. The system is designed to be robust against common tracking challenges like player occlusion and movement.

The core of the solution is a custom tracker that uses a Kalman Filter for motion prediction to ensure smooth and reliable tracking from frame to frame.

2. Features
Player Detection: Utilizes a pre-trained YOLO model to detect players in each frame.

Motion Prediction: Employs a Kalman Filter for each track to predict the player's position in the subsequent frame, making the tracking robust to brief obstructions.

Stable ID Assignment: Uses the Hungarian algorithm to optimally match predicted positions with new detections based on their proximity (Intersection over Union).

Track Lifecycle Management: A track is initialized in a "tentative" state (yellow box) and is promoted to "confirmed" (green box) only after being consistently tracked for several frames, preventing ghost tracks from false detections.

3. Setup and Installation
Follow these steps to set up the environment and run the project.

Prerequisites:

Python 3.8+

Git

Installation Steps:

Clone the repository:

git clone [https://github.com/TashonBraganca/Player-Tracker-YOLOv11.git](https://github.com/TashonBraganca/Player-Tracker-YOLOv11.git)
cd Player-Tracker-YOLOv11

Download the Model File:

IMPORTANT: The YOLO model file (best.pt) is too large for GitHub. Please download it from the original source.

Download Link: https://drive.google.com/file/d/1-5fOSHOSB9UXYP_enOoZNAMScrePVcMD/view

After downloading, place the best.pt file inside the data folder in the project directory.

Create and activate a virtual environment:

For Windows:

python -m venv venv
.\venv\Scripts\activate

For macOS/Linux:

python -m venv venv
source venv/bin/activate

Install the required dependencies from requirements.txt:

pip install -r requirements.txt
