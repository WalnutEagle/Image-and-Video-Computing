
# Group Members;-
#!) Samarth Singh(samarths@bu.edu) - U69593053
# 2) Adwait Kulkarni (adk1361@bu.edu)- U25712111
import cv2 as cv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_distance_matrix(objects_frame_n, objects_frame_np1):
    distance_matrix = np.zeros((len(objects_frame_n), len(objects_frame_np1)))
    for i, obj_n in enumerate(objects_frame_n):
        for j, obj_np1 in enumerate(objects_frame_np1):
            center_n = np.array(
                [obj_n['x_min'] + obj_n['width'] / 2, obj_n['y_min'] + obj_n['height'] / 2])
            center_np1 = np.array(
                [obj_np1['x_min'] + obj_np1['width'] / 2, obj_np1['y_min'] + obj_np1['height'] / 2])
            distance_matrix[i, j] = np.linalg.norm(center_n - center_np1)
    return distance_matrix


def assign_ids(frame_dict, max_distance=50, aspect_ratio_threshold=0.2):
    object_id_counter = 0
    prev_frame_objects = []
    object_history = {}  # New dictionary to track object history

    for frame, objects in frame_dict.items():
        current_frame_objects = objects
        if not prev_frame_objects:
            for obj in current_frame_objects:
                obj['id'] = object_id_counter
                object_history[object_id_counter] = [obj]  # Initialize history
                object_id_counter += 1
        else:
            unmatched_current_objs = set(range(len(current_frame_objects)))
            for i, obj_n in enumerate(prev_frame_objects):
                closest_dist = max_distance
                closest_j = None
                for j in unmatched_current_objs:
                    obj_np1 = current_frame_objects[j]
                    dist = np.linalg.norm([
                        obj_n['x_min'] + obj_n['width'] / 2 -
                        (obj_np1['x_min'] + obj_np1['width'] / 2),
                        obj_n['y_min'] + obj_n['height'] / 2 -
                        (obj_np1['y_min'] + obj_np1['height'] / 2)
                    ])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_j = j
                if closest_j is not None:
                    current_id = obj_n['id']
                    current_frame_objects[closest_j]['id'] = current_id
                    unmatched_current_objs.remove(closest_j)
                    object_history[current_id].append(
                        current_frame_objects[closest_j])  # Update history

            # Handle unmatched objects as potentially new
            for j in unmatched_current_objs:
                current_frame_objects[j]['id'] = object_id_counter
                # Initialize history for new objects
                object_history[object_id_counter] = [current_frame_objects[j]]
                object_id_counter += 1

        prev_frame_objects = [obj.copy() for obj in current_frame_objects]

    return frame_dict


def draw_object(object_dict, image, color=(0, 255, 0), thickness=2, c_color=(255, 0, 0)):
    x = object_dict['x_min']
    y = object_dict['y_min']
    width = object_dict['width']
    height = object_dict['height']
    cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)
    cv.putText(image, f"ID: {object_dict['id']}", (x,
               y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 2)
    return image


def draw_objects_in_video(video_file, frame_dict):
    cap = cv.VideoCapture(video_file)
    vidwrite = cv.VideoWriter(
        "part_2_demo_with_ids.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700, 500))
    count = 0
    ok, image = cap.read()
    while ok:
        image = cv.resize(image, (700, 500))
        if str(count) in frame_dict:
            obj_list = frame_dict[str(count)]
            for obj in obj_list:
                image = draw_object(obj, image)
        vidwrite.write(image)
        count += 1
        ok, image = cap.read()
    vidwrite.release()


# Assuming 'frame_dict.json' is loaded into 'frame_dict'
path_to_your_json_file = 'frame_dict.json'
with open(path_to_your_json_file, 'r') as file:
    frame_dict = json.load(file)

frame_dict_with_ids = assign_ids(frame_dict)
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file, frame_dict_with_ids)

with open('part_2_frame_dict.json', 'w') as file:
    json.dump(frame_dict_with_ids, file, indent=4)
