from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_tracks in tracks.values():
            for frame_tracks in object_tracks:
                for track_info in frame_tracks.values():
                    bbox = track_info['bbox']
                    if object_tracks is tracks['ball']:
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    track_info['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate and backfill missing values
        df_ball_positions.interpolate(inplace=True)
        df_ball_positions.bfill(inplace=True)

        return [{1: {"bbox": bbox}} for bbox in df_ball_positions.to_numpy().tolist()]

    def detect_frames(self, frames):
        batch_size = 20 
        detections = []
        for i in range(0, len(frames), batch_size):
            detections.extend(self.model.predict(frames[i:i + batch_size], conf=0.1))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names_inv = {v: k for k, v in detection.names.items()}

            # Convert to supervision format and handle "goalkeeper"
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_supervision.class_id = np.where(
                detection_supervision.class_id == cls_names_inv["goalkeeper"],
                cls_names_inv["player"],
                detection_supervision.class_id
            )

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for bbox, _, _, cls_id, track_id in detection_with_tracks:
                bbox = bbox.tolist()
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for bbox, _, _, cls_id in detection_supervision:
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_center = (x_center, y2 + 15)
            cv2.rectangle(
                frame,
                (rectangle_center[0] - 20, rectangle_center[1] - 10),
                (rectangle_center[0] + 20, rectangle_center[1] + 10),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (rectangle_center[0] - 8 if track_id > 99 else rectangle_center[0] - 12, rectangle_center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        team_1_frames = np.count_nonzero(team_ball_control[:frame_num + 1] == 1)
        team_2_frames = frame_num + 1 - team_1_frames

        team_1_control = team_1_frames / (team_1_frames + team_2_frames) * 100
        team_2_control = team_2_frames / (team_1_frames + team_2_frames) * 100

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_control:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_control:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame_copy = self.draw_ellipse(frame_copy, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame_copy = self.draw_triangle(frame_copy, player["bbox"], (0, 0, 255))

            for referee in tracks["referees"][frame_num].values():
                frame_copy = self.draw_ellipse(frame_copy, referee["bbox"], (0, 255, 255))

            for ball in tracks["ball"][frame_num].values():
                frame_copy = self.draw_triangle(frame_copy, ball["bbox"], (0, 255, 0))

            frame_copy = self.draw_team_ball_control(frame_copy, frame_num, team_ball_control)
            output_video_frames.append(frame_copy)

        return output_video_frames
