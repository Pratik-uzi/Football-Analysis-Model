from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape the image to 2D array and perform K-means with 2 clusters
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the top half of the player's bounding box
        top_half_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])][:, :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel and reshape to the image shape
        clustered_image = kmeans.labels_.reshape(top_half_image.shape[:2])

        # Identify player cluster based on the corners of the image
        corner_clusters = np.array([clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]])
        non_player_cluster = np.bincount(corner_clusters).argmax()
        player_cluster = 1 - non_player_cluster

        # Return the color corresponding to the player cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        # Collect colors for each player
        player_colors = [self.get_player_color(frame, player_detection["bbox"]) for player_detection in player_detections.values()]

        # Perform K-means clustering on player colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Store the resulting team colors and model
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Return the cached team assignment if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Determine player color and assign team
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Handle specific case for player ID 91
        if player_id == 91:
            team_id = 1

        # Cache and return the team assignment
        self.player_team_dict[player_id] = team_id
        return team_id

