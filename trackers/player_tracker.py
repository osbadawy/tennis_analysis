from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        
        for player_dict in player_detections:
            filtered_player_dict = {}
            for track_id, bbox in player_dict.items():
                if track_id in chosen_players:
                    # Rename the track_id to 1 or 2 based on the chosen_players mapping
                    new_id = chosen_players[track_id]
                    filtered_player_dict[new_id] = bbox
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        # Define the keypoint groups for each player
        player1_keypoints = [0, 4, 7, 1]
        player2_keypoints = [2, 5, 6, 3]
        
        # Initialize distances for both players
        player1_distances = []
        player2_distances = []
        
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            
            # Calculate minimum distance to player1 keypoints
            min_dist_p1 = float('inf')
            for kp_idx in player1_keypoints:
                kp_x = court_keypoints[kp_idx * 2]
                kp_y = court_keypoints[kp_idx * 2 + 1]
                distance = measure_distance(player_center, (kp_x, kp_y))
                if distance < min_dist_p1:
                    min_dist_p1 = distance
            player1_distances.append((track_id, min_dist_p1))
            
            # Calculate minimum distance to player2 keypoints
            min_dist_p2 = float('inf')
            for kp_idx in player2_keypoints:
                kp_x = court_keypoints[kp_idx * 2]
                kp_y = court_keypoints[kp_idx * 2 + 1]
                distance = measure_distance(player_center, (kp_x, kp_y))
                if distance < min_dist_p2:
                    min_dist_p2 = distance
            player2_distances.append((track_id, min_dist_p2))
        
        # Sort distances and choose players
        player1_distances.sort(key=lambda x: x[1])
        player2_distances.sort(key=lambda x: x[1])
        
        # Print distances for debugging
        print("\nPlayer Selection Distances:")
        print("Player 1 keypoints [0,4,7,1] distances:")
        for track_id, dist in player1_distances:
            print(f"Track ID {track_id}: {dist:.2f} pixels")
        
        print("\nPlayer 2 keypoints [2,5,6,3] distances:")
        for track_id, dist in player2_distances:
            print(f"Track ID {track_id}: {dist:.2f} pixels")
        
        # Create mapping from original track_id to new player id (1 or 2)
        chosen_players = {
            player1_distances[0][0]: 1,  # Player closest to player1 keypoints becomes player 1
            player2_distances[0][0]: 2   # Player closest to player2 keypoints becomes player 2
        }
        
        print("\nSelected Players:")
        print(f"Player 1: Track ID {player1_distances[0][0]} (distance: {player1_distances[0][1]:.2f})")
        print(f"Player 2: Track ID {player2_distances[0][0]} (distance: {player2_distances[0][1]:.2f})")
        
        return chosen_players

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                # Ensure all coordinates are integers
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    