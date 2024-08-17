import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_dimensions = (68, 23.32)  # (width, length)
        
        self.pixel_vertices = np.array([[110, 1035], 
                                        [265, 275], 
                                        [910, 260], 
                                        [1640, 915]], dtype=np.float32)
        
        self.target_vertices = np.array([
            [0, court_dimensions[0]],
            [0, 0],
            [court_dimensions[1], 0],
            [court_dimensions[1], court_dimensions[0]]
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = tuple(map(int, point))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None
        
        reshaped_point = np.float32([point]).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_tracks in tracks.values():
            for frame_num, track in enumerate(object_tracks):
                for track_info in track.values():
                    position = np.array(track_info['position_adjusted'])
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        track_info['position_transformed'] = transformed_position.squeeze().tolist()
                    else:
                        track_info['position_transformed'] = None
