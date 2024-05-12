import cv2
import numpy as np
import sys


def convert_to_hsv_and_detect_skin(src):
    """Converts an RGB image to HSV and detects skin based on HSV thresholds."""
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return skin_mask


def my_frame_differencing(prev, curr):
    """Performs frame differencing between two consecutive frames."""
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, dst = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return dst


def my_motion_energy(mh):
    """Calculates motion energy by accumulating frame differences."""
    dst = np.max(np.stack(mh, axis=0), axis=0).astype(np.uint8)
    return dst


def adjust_thresholds_based_on_conditions(current_frame, skin_mask):
    """Dynamically adjusts thresholds based on lighting conditions and skin area."""
    average_brightness = np.mean(current_frame)
    skin_area = np.count_nonzero(skin_mask)
    thresholds = {"swipe_threshold": 0.20, "circle_detection_sensitivity": 0.1}
    if average_brightness < 50:
        thresholds["swipe_threshold"] += 0.05
    if skin_area > 50000:
        thresholds["circle_detection_sensitivity"] += 0.05
    return thresholds


def analyze_motion_pattern(motion_energy, frame_shape):
    """Analyzes the motion pattern to recognize complex gestures. Placeholder for analysis logic."""
    # Example placeholder logic
    return "No Gesture Detected"


def recognize_gesture(skin_mask, motion_energy, current_frame):
    thresholds = adjust_thresholds_based_on_conditions(current_frame, skin_mask)
    swipe_threshold = thresholds["swipe_threshold"] * current_frame.shape[1]  # Convert percentage to pixels
    
    skin_area = cv2.countNonZero(skin_mask)
    if skin_area < 10000:  # Threshold for minimal skin area to consider a gesture
        return "No Gesture"
    
    M = cv2.moments(motion_energy)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # Centroid X
        cy = int(M["m01"] / M["m00"])  # Centroid Y
        center_x = current_frame.shape[1] // 2
        center_y = current_frame.shape[0] // 2  # Calculate the center Y of the frame
        
        # Detect horizontal movement
        if cx < center_x - swipe_threshold:
            return "Swipe Left"
        elif cx > center_x + swipe_threshold:
            return "Swipe Right"
        
        # Detect vertical movement (rudimentary approach)
        # Assuming a simplistic detection for demonstration purposes
        vertical_movement_threshold = current_frame.shape[0] * 0.1  # 10% of frame height
        if cy < center_y - vertical_movement_threshold:
            return "Hand Movement Up"
        elif cy > center_y + vertical_movement_threshold:
            return "Hand Movement Down"
        
        return analyze_motion_pattern(motion_energy, current_frame.shape)
    
    return "Gesture Not Recognized"


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the video cam")
        sys.exit()
    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read a frame from video stream")
        sys.exit()
    len_history = 7
    my_motion_history = [np.zeros(frame0.shape[:2], dtype=np.uint8)
                         for _ in range(len_history)]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read a frame from video stream")
            break
        skin_mask = convert_to_hsv_and_detect_skin(frame)
        frame_diff = my_frame_differencing(frame0, frame)
        my_motion_history.pop(0)
        my_motion_history.append(frame_diff)
        motion_energy = my_motion_energy(my_motion_history)
        gesture = recognize_gesture(skin_mask, motion_energy, frame)
        cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("MyVideo0", frame)
        frame0 = frame
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()