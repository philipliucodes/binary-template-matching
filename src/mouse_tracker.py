import math
import numpy as np
import cv2
import pandas as pd
import template_matcher
import os
import glob
import matplotlib.pyplot as plt

def extract_frames(video_path, white_threshold, start_time, end_time):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    interval = int(0.1 * fps)  # 100ms interval
    
    rgb_frames = [] # List to store rgb frames
    binary_frames = []  # List to store binary frames
    current_frame = start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    while current_frame <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frames.append(frame) # bgr
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_frame = template_matcher.binarize_image(gray_frame, white_threshold)
        binary_frames.append(binary_frame)
        current_frame += interval
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    video_capture.release()
    return np.array(binary_frames), np.array(rgb_frames)  # Return an array of grayscale frames

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

def pick_pixel(frame):
    selected_pixel = (-1, -1)
    def on_click(event):
        global selected_pixel
        if event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if 0 <= row < 768 and 0 <= col < 1280:
                selected_pixel = (row, col)
            plt.close()
    fig, ax = plt.subplots()
    ax.imshow(frame)
    fig.canvas.mpl_connect('button_press_event', on_click)
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.show()
    return selected_pixel

def get_cursor_loc(binary_frames, rgb_frames, idx, templates, prev_x, prev_y, detection_threshold):
    frame = binary_frames[idx]
    detected_regions = []
    while detection_threshold >= 0.7:
        for template_path, template in templates:
            w, h = template.shape[::-1]
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= detection_threshold)

            for pt in zip(*locations[::-1]):
                detected_regions.append((int(pt[0]), int(pt[1])))
                cv2.rectangle(rgb_frames[idx], pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
        if(len(detected_regions) != 0):
            break
        detection_threshold -= 0.05
    if len(detected_regions) == 1: # one
        return detected_regions[0][0], detected_regions[0][1]
    elif len(detected_regions) > 10 or len(detected_regions) == 0: # large multiple
        curr_x, curr_y = pick_pixel(rgb_frames[idx])
        if curr_x != -1 and curr_y != -1:
            return curr_x, curr_y
        else:
            return prev_x, prev_y
    else: # if multiple, check if regions are close together (within 3 pixels)
        # if close, return average
        close = True
        for idx1 in range(len(detected_regions)):
            region = detected_regions[idx1]
            for idx2 in range(idx1, len(detected_regions)):
                if int(round(math.sqrt((region[idx1] - region[idx2])**2 + (region[idx1] - region[idx2])**2))) > 10:
                    close = False
                    break
            if close == False:
                break
        if close:
            x_list = [region[0] for region in detected_regions]
            y_list = [region[1] for region in detected_regions]
            return (int(round(np.mean(x_list))), int(round(np.mean(y_list))))
        else:
            print(len(detected_regions))
            curr_x, curr_y = pick_pixel(binary_frames[idx])
            curr_x, curr_y = pick_pixel(rgb_frames[idx])
            if curr_x != -1 and curr_y != -1:
                return curr_x, curr_y
            else:
                return prev_x, prev_y

def make_tr_dict(binary_frames, rgb_frames, templates, prev_x, prev_y, detection_threshold):
    tr_dist_dict = {}
    # if only 1 match, then get x and y, if none, decrease threshold until 1 match, if multiple matches then discard/infer from previous
    frame_counter = 1
    #for tr in range(600): # 15*60/1.5 = 600 TR
    for tr in range(20): # TESTING
        dists = []
        for ms in range(15): # 15 units of 100ms in each TR (NEED TO CHANGE IF DECREASE INTERVAL)
            curr_x, curr_y = get_cursor_loc(binary_frames, rgb_frames, frame_counter, templates, prev_x, prev_y, detection_threshold)
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

def main(input_filepath, output_filepath, templates_folderpath, start_time, end_time):
    # paths
    if not os.path.exists(input_filepath):
        print(f"File '{input_filepath}' doesn't exist.")
        exit()
    if os.path.exists(output_filepath):
        print(f"File '{output_filepath}' exists.")
        exit()
    if not os.path.exists(templates_folderpath):
        print(f"File '{templates_folderpath}' doesn't exist.")
        exit()
    
    # thresholds
    detection_threshold = 0.90
    white_threshold = 207

    # extract frames
    binary_frames, rgb_frames = extract_frames(input_filepath, white_threshold, start_time, end_time)

    # get templates
    templates = load_templates(templates_folderpath, white_threshold)

    # get frame dimensions
    dim_y, dim_x = binary_frames[0].shape

    # get initial frame
    prev_x, prev_y = get_cursor_loc(binary_frames, rgb_frames, 0, templates, int(round(dim_x/2)), int(round(dim_y/2)), detection_threshold)

    # make dict
    tr_dist_dict = make_tr_dict(binary_frames, rgb_frames, templates, prev_x, prev_y, detection_threshold)

    # save dict
    dict_to_csv(tr_dist_dict, output_filepath)

if __name__ == "__main__":
    # input_filepath = 'Web47_run1.mkv'
    # output_filepath = 'Web47_1_mouse_dists_selection.csv'
    # templates_folderpath = 'binary-image-matching/src/templates'
    # start_time = 25
    # end_time = 925
    # main(input_filepath, output_filepath, templates_folderpath, start_time, end_time)

    input_filepath = 'Web08_run2.mkv'
    output_filepath = 'Web08_2_mouse_dists_selection.csv'
    templates_folderpath = 'binary-image-matching/src/templates'
    start_time = 185
    end_time = 1085
    main(input_filepath, output_filepath, templates_folderpath, start_time, 215)