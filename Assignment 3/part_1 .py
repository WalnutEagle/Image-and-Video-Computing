#Group Members;-
#!) Samarth Singh(samarths@bu.edu) - U69593053
#2) Adwait Kulkarni (adk1361@bu.edu)- U25712111


import json
import cv2 as cv
import numpy as np

# Kalman Filter implementation


class KalmanFilter:
    def __init__(self, dt, process_noise_std, measurement_noise_std, initial_state):
        self.dt = dt  # Time step

        # State transition model
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Observation model
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Process noise covariance
        self.Q = np.eye(4) * process_noise_std**2
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise_std**2
        # Initial state
        self.x = np.array(initial_state)
        # Initial covariance matrix
        self.P = np.eye(4)

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Function to load object tracking data from a file


def load_obj_each_frame(data_file):
    with open(data_file, 'r') as file:
        frame_dict = json.load(file)
    return frame_dict

# Function to draw target object center using Kalman filter estimates

def save_path_points(path_points, output_file="path_points.json"):
    with open(output_file, 'w') as file:
        json.dump(path_points, file)

def draw_target_object_center_kf_with_path(video_file, obj_centers, output_path="part_1(Solution).mp4"):
    # Initialize the Kalman filter with the first valid observation
    for observation in obj_centers:
        if observation != [-1, -1]:
            initial_state = [observation[0],
                             observation[1], 0, 0]  # [x, y, dx, dy]
            break
    kf = KalmanFilter(dt=1/30, process_noise_std=1,
                      measurement_noise_std=10, initial_state=initial_state)

    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    if not ok:
        print("Failed to read video file.")
        return

    # Create video writer
    vidwrite = cv.VideoWriter(
        output_path, cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))

    # To accumulate path points
    path_points = []

    while ok:
        image = cv.resize(image, (700, 500))  # Resize to match the coords

        # Predict the next state
        kf.predict()

        # Get the current observation, update if it's not a missing observation
        if obj_centers:
            current_observation = obj_centers.pop(0)
            if current_observation != [-1, -1]:
                kf.update(current_observation)

        # Add current position to the path
        pos_x, pos_y = kf.x[0], kf.x[1]
        path_points.append((int(pos_x), int(pos_y)))

        # Draw the entire path
        for i in range(1, len(path_points)):
            if path_points[i - 1] is None or path_points[i] is None:
                continue
            cv.line(image, path_points[i - 1], path_points[i], (0, 0, 255), 2)

        vidwrite.write(image)
        ok, image = cap.read()

    vidwrite.release()
    save_path_points(path_points)


# Load the observations and path to the video file
frame_dict = load_obj_each_frame("adwait/object_to_track.json")
video_file = "adwait/commonwealth copy.mp4"

# Apply the drawing function with Kalman filter
draw_target_object_center_kf_with_path(
    video_file, frame_dict['obj'], "part_1(Solution).mp4")
