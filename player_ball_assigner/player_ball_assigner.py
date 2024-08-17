import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self, max_distance=70):
        self.max_player_ball_distance = max_distance
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Measure distances from both corners at the bottom of the bounding box
            distances = [
                measure_distance((player_bbox[0], player_bbox[-1]), ball_position),
                measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            ]
            
            distance = min(distances)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player
