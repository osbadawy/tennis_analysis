import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from trackers.player_tracker import PlayerTracker

@pytest.fixture
def player_tracker():
    # Create a minimal tracker instance without model initialization
    tracker = PlayerTracker.__new__(PlayerTracker)
    return tracker

def test_choose_players(player_tracker):
    # Test data based on real example
    court_keypoints = np.array([
        649.3417, 273.86682,  # Keypoint 0
        1259.1753, 274.36398,  # Keypoint 1
        414.30038, 797.74927,  # Keypoint 2
        1472.5964, 799.99615,  # Keypoint 3
        726.1677, 273.8491,    # Keypoint 4
        546.3561, 797.9864,    # Keypoint 5
        1182.9243, 274.2681,   # Keypoint 6
        1339.3441, 799.63385,  # Keypoint 7
        699.25854, 349.78043,  # Keypoint 8
        1206.1588, 350.28195,  # Keypoint 9
        607.1354, 615.6376,    # Keypoint 10
        1286.2499, 616.84314,  # Keypoint 11
        952.8101, 349.9359,    # Keypoint 12
        945.7651, 616.2498     # Keypoint 13
    ], dtype=np.float32)

    player_dict = {
        1: [1024.6578, 147.7072, 1068.7899, 272.3576],  # Player near top of court
        2: [1219.9573, 966.6572, 1311.2650, 1042.9421],  # Player near bottom right
        4: [1113.9965, 980.6678, 1197.5819, 1041.9597],  # Another player near bottom
        5: [1369.0159, 101.3881, 1392.9397, 189.3555],   # Player near top right
        7: [443.0236, 806.1839, 505.0568, 907.4980],     # Player near bottom left
        8: [1514.1956, 1010.7090, 1615.7367, 1079.9553], # Player far right
        9: [295.4647, 387.0532, 349.1214, 452.6509],     # Player middle left
        10: [720.1429, 985.2456, 800.0311, 1079.6555]    # Player bottom middle
    }

    # Call the method
    chosen_players = player_tracker.choose_players(court_keypoints, player_dict)

    # Assertions based on actual debug output
    assert len(chosen_players) == 2
    # Player 1 should be track ID 7 (closest to player1 keypoints)
    assert chosen_players[7] == 1
    # Player 2 should be track ID 1 (closest to player2 keypoints)
    assert chosen_players[1] == 2

def test_choose_players_with_empty_dict(player_tracker):
    # Test with empty player dictionary
    court_keypoints = np.array([
        649.3417, 273.86682,
        1259.1753, 274.36398,
        414.30038, 797.74927,
        1472.5964, 799.99615
    ], dtype=np.float32)
    player_dict = {}
    
    chosen_players = player_tracker.choose_players(court_keypoints, player_dict)
    assert len(chosen_players) == 0

def test_choose_and_filter_players(player_tracker):
    # Test data based on real example
    court_keypoints = np.array([
        649.3417, 273.86682,
        1259.1753, 274.36398,
        414.30038, 797.74927,
        1472.5964, 799.99615
    ], dtype=np.float32)

    player_detections = [
        {7: [443.0236, 806.1839, 505.0568, 907.4980],    # Player 1 (bottom left)
         1: [1024.6578, 147.7072, 1068.7899, 272.3576]}, # Player 2 (top)
        {7: [444.0236, 807.1839, 506.0568, 908.4980],    # Player 1 (bottom left)
         1: [1025.6578, 148.7072, 1069.7899, 273.3576]}, # Player 2 (top)
        {7: [445.0236, 808.1839, 507.0568, 909.4980],    # Player 1 (bottom left)
         1: [1026.6578, 149.7072, 1070.7899, 274.3576]}  # Player 2 (top)
    ]

    # Mock choose_players to return specific mapping based on actual behavior
    with patch.object(PlayerTracker, 'choose_players', return_value={7: 1, 1: 2}):
        filtered_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

        # Assertions
        assert len(filtered_detections) == 3
        
        # Check first frame
        assert len(filtered_detections[0]) == 2
        assert filtered_detections[0][1] == [443.0236, 806.1839, 505.0568, 907.4980]  # Player 1 (bottom)
        assert filtered_detections[0][2] == [1024.6578, 147.7072, 1068.7899, 272.3576]  # Player 2 (top)
        
        # Check second frame
        assert len(filtered_detections[1]) == 2
        assert filtered_detections[1][1] == [444.0236, 807.1839, 506.0568, 908.4980]  # Player 1 (bottom)
        assert filtered_detections[1][2] == [1025.6578, 148.7072, 1069.7899, 273.3576]  # Player 2 (top)
        
        # Check third frame
        assert len(filtered_detections[2]) == 2
        assert filtered_detections[2][1] == [445.0236, 808.1839, 507.0568, 909.4980]  # Player 1 (bottom)
        assert filtered_detections[2][2] == [1026.6578, 149.7072, 1070.7899, 274.3576]  # Player 2 (top)

def test_choose_and_filter_players_with_empty_detections(player_tracker):
    # Test with empty detections
    court_keypoints = np.array([
        649.3417, 273.86682,
        1259.1753, 274.36398,
        414.30038, 797.74927,
        1472.5964, 799.99615
    ], dtype=np.float32)
    player_detections = []
    
    # Mock choose_players to return empty dict
    with patch.object(PlayerTracker, 'choose_players', return_value={}):
        filtered_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        assert len(filtered_detections) == 0

def test_choose_and_filter_players_with_single_frame(player_tracker):
    # Test with single frame
    court_keypoints = np.array([
        649.3417, 273.86682,
        1259.1753, 274.36398,
        414.30038, 797.74927,
        1472.5964, 799.99615
    ], dtype=np.float32)
    
    player_detections = [
        {7: [443.0236, 806.1839, 505.0568, 907.4980],    # Player 1 (bottom left)
         1: [1024.6578, 147.7072, 1068.7899, 272.3576]}  # Player 2 (top)
    ]
    
    # Mock choose_players to return specific mapping based on actual behavior
    with patch.object(PlayerTracker, 'choose_players', return_value={7: 1, 1: 2}):
        filtered_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        
        # Assertions
        assert len(filtered_detections) == 1
        assert len(filtered_detections[0]) == 2
        assert filtered_detections[0][1] == [443.0236, 806.1839, 505.0568, 907.4980]  # Player 1 (bottom)
        assert filtered_detections[0][2] == [1024.6578, 147.7072, 1068.7899, 272.3576]  # Player 2 (top) 