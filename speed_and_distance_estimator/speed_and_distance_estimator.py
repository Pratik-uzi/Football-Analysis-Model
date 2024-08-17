import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator:
    def __init__(self, frame_window=5, frame_rate=24):
        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object in {"ball", "referees"}:
                continue

            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_kmh = (distance_covered / time_elapsed) * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id in tracks[object][frame_num_batch]:
                            track_info = tracks[object][frame_num_batch][track_id]
                            track_info['speed'] = speed_kmh
                            track_info['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object in {"ball", "referees"}:
                    continue

                for track_info in object_tracks[frame_num].values():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')

                    if speed is not None and distance is not None:
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = (int(position[0]), int(position[1] + 40))

                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
