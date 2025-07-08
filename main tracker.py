import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

model_path = os.path.join('data', 'best.pt')
video_path = os.path.join('data', '15sec_input_720p.mp4')
output_path = 'final video.mp4'

class Track:
    def __init__(self, id, box, feature):
        self.id = id
        self.features = deque([feature], maxlen=50)
        self.hits = 1
        self.frames_since_update = 0
        self.age = 0
        self.state = 'tentative'

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,1], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]], dtype=np.float32)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.box_to_z(box)

    @property
    def feature(self):
        return np.mean(self.features, axis=0)

    def box_to_z(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = box[0] + w / 2.
        y = box[1] + h / 2.
        a = w / h if h > 0 else 0
        return np.array([x, y, a, h], dtype=np.float32).reshape((4, 1))

    def z_to_box(self):
        state = self.kf.x.flatten()
        w = state[2] * state[3]
        h = state[3]
        return np.array([
            state[0] - w / 2., state[1] - h / 2.,
            state[0] + w / 2., state[1] + h / 2.
        ], dtype=np.int32)

    @property
    def box(self):
        return self.z_to_box()

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.frames_since_update += 1

    def update(self, box, feature, min_hits):
        self.kf.update(self.box_to_z(box))
        self.features.append(feature)
        self.hits += 1
        self.frames_since_update = 0
        if self.state == 'tentative' and self.hits >= min_hits:
            self.state = 'confirmed'

class PlayerTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def get_features(self, frame, boxes):
        features = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 >= x2 or y1 >= y2:
                features.append(np.zeros(8 * 8 * 8))
                continue
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            features.append(hist.flatten())
        return np.array(features)

    def calculate_iou_cost(self, tracks, boxes):
        if not tracks or len(boxes) == 0:
            return np.empty((len(tracks), 0))
        
        track_boxes = np.array([track.box for track in tracks])
        iou_matrix = np.zeros((len(tracks), len(boxes)), dtype=np.float32)
        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(boxes):
                x1, y1, x2, y2 = track_box
                x1_d, y1_d, x2_d, y2_d = det_box
                inter_x1 = max(x1, x1_d)
                inter_y1 = max(y1, y1_d)
                inter_x2 = min(x2, x2_d)
                inter_y2 = min(y2, y2_d)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                track_area = (x2 - x1) * (y2 - y1)
                det_area = (x2_d - x1_d) * (y2_d - y1_d)
                union_area = track_area + det_area - inter_area
                iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0
        return 1 - iou_matrix

    def update(self, frame, boxes):
        for track in self.tracks:
            track.predict()

        detection_features = self.get_features(frame, boxes)

        cost_matrix = self.calculate_iou_cost(self.tracks, boxes)
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()
        for t_idx, d_idx in zip(track_indices, detection_indices):
            if cost_matrix[t_idx, d_idx] > (1 - self.iou_threshold):
                continue
            self.tracks[t_idx].update(boxes[d_idx], detection_features[d_idx], self.min_hits)
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)
        
        for i in range(len(boxes)):
            if i not in matched_detections:
                self.create_track(boxes[i], detection_features[i])

        self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_age]
        return self.tracks

    def create_track(self, box, feature):
        self.tracks.append(Track(self.next_id, box, feature))
        self.next_id += 1

def main():
    model = YOLO(model_path)
    tracker = PlayerTracker(min_hits=3, iou_threshold=0.3)

    player_class = [i for i, name in model.names.items() if name.lower() == 'player'][0]
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        print(f"Processing frame {frame_num}...")

        results = model(frame, verbose=False, conf=0.5, classes=[player_class])
        player_boxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes]

        all_tracks = tracker.update(frame, player_boxes)

        for track in all_tracks:
            if track.frames_since_update == 0:
                if track.state == 'confirmed':
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                
                x1, y1, x2, y2 = track.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track.id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        writer.write(frame)