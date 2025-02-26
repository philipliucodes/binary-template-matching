import math
import numpy as np
import cv2
import pandas as pd
import template_matcher
import os
import glob

def extract_frames(video_path, start_time, end_time):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    interval = int(0.1 * fps)  # 100ms interval
    
    grayscale_frames = []  # List to store grayscale frames
    current_frame = start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    while current_frame <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_frame = template_matcher.binarize_image(gray_frame)
        grayscale_frames.append(binary_frame)
        current_frame += interval
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    video_capture.release()
    return np.array(grayscale_frames)  # Return an array of grayscale frames

def closest_point(point, pt_list):
    x, y = point
    dists = []
    for pt_x, pt_y in pt_list:
        dists.append(int(round(math.sqrt((pt_x - x)**2 + (pt_y - y)**2))))
    return pt_list[dists.index(min(dists))]

def load_templates(templates_folder, white_threshold):
    images = []
    image_patterns = ["*.jp*g", "*.png", "*.bmp"]

    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(templates_folder, pattern)))

    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append((img_path, image))
        else:
            print(f"Warning: Failed to load image '{img_path}'.")
    if not images:
        print("Error: No valid template images loaded.")
    return [(path, template_matcher.binarize_image(img, white_threshold)) for path, img in images]

def get_cursor_loc(frames, idx, templates, prev_x, prev_y, detection_threshold, break_threshold, detection_threshold_decrement):
    frame = frames[idx]
    detected_regions = []
    while not detected_regions:
        for template_path, template in templates:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= detection_threshold)

            for pt in zip(*locations[::-1]):
                # w, h = template.shape[::-1]
                # detected_regions.append((pt[0], pt[1], w, h, template_path))
                detected_regions.append((int(pt[0]), int(pt[1])))
        if detection_threshold <= break_threshold:
            break
        detection_threshold -= detection_threshold_decrement
    if len(detected_regions) == 1: # one
        return detected_regions[0][0], detected_regions[0][1]
    elif not detected_regions: # none
        return prev_x, prev_y
    else: # more than one -> pick the closest
        return closest_point((prev_x, prev_y), detected_regions)

def make_tr_dict(frames, templates, prev_x, prev_y, detection_threshold, break_threshold, detection_threshold_decrement):
    tr_dist_dict = {}
    # if only 1 match, then get x and y, if none, decrease threshold until 1 match, if multiple matches then discard/infer from previous
    frame_counter = 1
    for tr in range(600): # 15*60/1.5 = 600 TR
        dists = []
        for ms in range(15): # 15 units of 100ms in each TR
            curr_x, curr_y = get_cursor_loc(frames, frame_counter, templates, prev_x, prev_y, detection_threshold, break_threshold, detection_threshold_decrement)
            dists.append(int(round(math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2))))
            prev_x, prev_y = curr_x, curr_y
            frame_counter += 1

        # store the data by hashmapping by TR (like 1: (sum of dists for 15 intervals each interval being 100ms))
        tr_dist_dict[tr+1] = sum(dists)
    return tr_dist_dict

def dict_to_csv(tr_dist_dict, output_filepath):
    df = pd.DataFrame.from_dict(tr_dist_dict, orient='index', columns=["Distance"])
    df.index.name = "TR"
    df.to_csv(output_filepath)

def main():
    # paths
    input_filepath = 'Web47_run1.mkv'
    if not os.path.exists(input_filepath):
        print(f"File '{input_filepath}' doesn't exist.")
        exit()
    output_filepath = 'tr_dist_dict_10_50_new.csv'
    if os.path.exists(output_filepath):
        print(f"File '{output_filepath}' exists.")
        exit()
    templates_folderpath = 'templates'
    if not os.path.exists(templates_folderpath):
        print(f"File '{templates_folderpath}' doesn't exist.")
        exit()
    
    # thresholds
    detection_threshold = 0.9
    detection_threshold_decrement = 0.025
    break_threshold = 0.8
    white_threshold = 200

    # extract frames
    frames = extract_frames(input_filepath, 25, 925)

    # get templates
    templates = load_templates(templates_folderpath, white_threshold)

    # get initial frame
    prev_x, prev_y = get_cursor_loc(frames, 0, templates, 750, 400, detection_threshold, break_threshold, detection_threshold_decrement)

    # make dict
    tr_dist_dict = make_tr_dict(frames, templates, prev_x, prev_y, detection_threshold, break_threshold, detection_threshold_decrement)

    # save dict
    dict_to_csv(tr_dist_dict, output_filepath)

if __name__ == "__main__":
    main()