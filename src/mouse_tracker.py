import os
import argparse
import numpy as np
import csv
from PIL import Image
import cv2
import ffmpeg
import queue
import threading

frame_review_queue = queue.Queue()
processing_done = threading.Event()

def is_image(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def get_video_duration(video_path):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return duration

def transform_pixels(image_array, alpha_channel, white_threshold):
    transformed = np.zeros(image_array.shape[:2], dtype=np.uint8)
    non_transparent = alpha_channel > 0
    almost_white = (
        (image_array[:, :, 0] > white_threshold) &
        (image_array[:, :, 1] > white_threshold) &
        (image_array[:, :, 2] > white_threshold)
    )
    transformed[almost_white & non_transparent] = 255
    transformed[~almost_white & non_transparent] = 0
    return transformed

def extract_frames(video_path, white_threshold, start_frame, end_frame, interval, fps):
    video_capture = cv2.VideoCapture(video_path)
    rgb_frames = []
    timestamps = []
    current_frame = start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    while current_frame <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frames.append(frame)
        current_time = current_frame / fps
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        milliseconds = int(round((current_time - int(current_time)) * 1000))
        timestamp = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
        timestamps.append(timestamp)
        current_frame += interval
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    video_capture.release()
    return rgb_frames, timestamps

def overwrite_csv_row(csv_path, timestamp, new_row):
    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    updated = False
    for i in range(1, len(rows)):
        if rows[i][0] == timestamp:
            rows[i] = new_row
            updated = True
            break
    if updated:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    else:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

def process_frame(frame_array, timestamp, template_images, confidence_threshold, white_threshold, csv_output, 
                  last_matched_template, last_match_position, search_width, search_height, frame_index, use_limited_templates):
    input_image = Image.fromarray(frame_array).convert("RGBA")
    input_array = np.array(input_image)
    input_alpha = input_array[:, :, 3]
    input_transformed = transform_pixels(input_array, input_alpha, white_threshold)

    if use_limited_templates:
        candidate_templates = []

        use_last_template = (frame_index % 2 == 0)

        if use_last_template and last_matched_template:
            for tmpl in template_images:
                if os.path.basename(tmpl) == last_matched_template:
                    candidate_templates.append(tmpl)
                    break
        else:
            if os.path.basename(template_images[0]) != last_matched_template:
                candidate_templates.append(template_images[0])
            elif len(template_images) > 1:
                candidate_templates.append(template_images[1])
    else:
        if last_matched_template:
            candidate_templates = []
            for tmpl in template_images:
                if os.path.basename(tmpl) == last_matched_template:
                    candidate_templates.append(tmpl)
                    break
            candidate_templates += [t for t in template_images if os.path.basename(t) != last_matched_template]
        else:
            candidate_templates = template_images

    match_found = False
    matched_template = None
    match_position = (None, None)
    match_percentage = 0.0

    for template_filename in candidate_templates:
        template_image = Image.open(template_filename).convert("RGBA")
        template_array = np.array(template_image)
        template_alpha = template_array[:, :, 3]
        template_transformed = transform_pixels(template_array, template_alpha, white_threshold)

        ih, iw = input_transformed.shape
        th, tw = template_transformed.shape

        search_regions = []
        if last_match_position is not None:
            x_prev, y_prev = last_match_position
            x_start = max(0, x_prev - search_width // 2)
            x_end = min(iw - tw, x_prev + search_width // 2)
            y_start = max(0, y_prev - search_height // 2)
            y_end = min(ih - th, y_prev + search_height // 2)
            search_regions.append((x_start, x_end, y_start, y_end))

        search_regions.append((0, iw - tw, 0, ih - th))

        for x_start, x_end, y_start, y_end in search_regions:
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    roi = input_transformed[y:y+th, x:x+tw]
                    mask = template_alpha > 0
                    total_pixels = np.count_nonzero(mask)

                    if total_pixels > 0:
                        matching_pixels = np.sum(roi[mask] == template_transformed[mask])
                        match_score = matching_pixels / total_pixels

                        if match_score > confidence_threshold:
                            matched_template = os.path.basename(template_filename)
                            match_position = (x, y)
                            match_percentage = match_score * 100
                            match_found = True
                            break
                if match_found:
                    break
            if match_found:
                break
        if match_found:
            break

    with open(csv_output, mode='a', newline='') as file:
        writer = csv.writer(file)
        if match_found:
            writer.writerow([timestamp, matched_template, match_position[0], match_position[1], f"{match_percentage:.2f}"])
            print(f"[{timestamp}] Matched '{matched_template}' at ({match_position[0]}, {match_position[1]}) with {match_percentage:.2f}% confidence.")
            return matched_template, match_position
        else:
            writer.writerow([timestamp, "No match", "N/A", "N/A", "N/A"])
            print(f"[{timestamp}] No template match found. Awaiting user input...")
            frame_review_queue.put(timestamp)
            return None, None

def manual_review_loop(csv_output, video_path):
    screen_res = 1920, 1080
    scale_percent = 200
    last_manual_coords = None

    while True:
        if processing_done.is_set() and frame_review_queue.empty():
            break

        if not frame_review_queue.empty():
            timestamp = frame_review_queue.get()

            # Convert timestamp back to frame number
            try:
                parts = list(map(int, timestamp.split('_')))
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int((parts[0] * 60 + parts[1]) * fps + parts[2] * fps / 1000)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    print(f"[ERROR] Could not retrieve frame for timestamp {timestamp}")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to load frame for timestamp {timestamp}: {e}")
                continue

            window_name = f"Manual Review - {timestamp}"
            clicked_coords = []
            clicked = [False]
            object_not_present = [False]
            reuse_last_coords = [False]

            height, width = frame.shape[:2]
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            def click_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and not clicked[0] and not object_not_present[0]:
                    orig_x = int(x * width / new_width)
                    orig_y = int(y * height / new_height)
                    clicked_coords.append((orig_x, orig_y))
                    clicked[0] = True
                    print(f"[MANUAL INPUT] User clicked at ({orig_x}, {orig_y}) on frame {timestamp}")
                    cv2.setMouseCallback(window_name, lambda *args: None)
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow(window_name)

            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, new_width, new_height)
                cv2.moveWindow(window_name, screen_res[0] // 2 - new_width // 2, screen_res[1] // 2 - new_height // 2)
                cv2.imshow(window_name, resized_frame)
                cv2.setMouseCallback(window_name, click_callback)
            except Exception as e:
                print(f"[ERROR] Could not create window for manual review: {e}")
                continue

            while True:
                key = cv2.waitKey(10)

                if key == ord('a'):
                    print(f"[MANUAL INPUT] 'a' pressed on frame {timestamp}. Object not present.")
                    object_not_present[0] = True
                    break

                elif key == ord('s'):
                    if last_manual_coords:
                        clicked_coords.append(last_manual_coords)
                        reuse_last_coords[0] = True
                        print(f"[MANUAL INPUT] 's' pressed on frame {timestamp}. Reusing coordinates {last_manual_coords}")
                    else:
                        print(f"[MANUAL INPUT] 's' pressed but no previous coordinates available. Marking as not present.")
                        object_not_present[0] = True
                    break

                if clicked[0] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_name)

            if object_not_present[0]:
                overwrite_csv_row(csv_output, timestamp, [timestamp, "Not present", "N/A", "N/A", "N/A"])

            elif reuse_last_coords[0]:
                x, y = last_manual_coords
                overwrite_csv_row(csv_output, timestamp, [timestamp, "Manual selection", x, y, "N/A"])

            elif not clicked_coords:
                print(f"[MANUAL INPUT] Window closed without a click for frame {timestamp}")
                overwrite_csv_row(csv_output, timestamp, [timestamp, "Not present", "N/A", "N/A", "N/A"])

            else:
                x, y = clicked_coords[0]
                last_manual_coords = (x, y)
                overwrite_csv_row(csv_output, timestamp, [timestamp, "Manual selection", x, y, "N/A"])
        else:
            cv2.waitKey(50)

    print("[INFO] All frames processed and manual review complete. Exiting...")

def template_matcher(video_path, template_path, interval_sec, confidence_threshold, white_threshold, output_dir,
                     search_width, search_height, batch_size, start_time, end_time):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_output = os.path.join(output_dir, f"{video_name}_match_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

    if interval_sec * fps < 1:
        print(f"[ERROR] The interval ({interval_sec}s) is too small for the video's FPS ({fps}). Please use a larger interval.")
        processing_done.set()
        return

    interval = max(1, int(interval_sec * fps))
    total_frames = int(duration * fps)

    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames - 1
    end_frame = min(end_frame, total_frames - 1)

    template_images = [os.path.join(template_path, f) for f in sorted(os.listdir(template_path)) if is_image(f)]
    if not template_images:
        print(f"[ERROR] No valid template images found in '{template_path}'")
        processing_done.set()
        return

    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "best_template", "match_x", "match_y", "match_percentage"])

    last_matched_template = None
    last_match_position = None
    frame_counter = 0

    for batch_start in range(start_frame, end_frame + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_frame)
        rgb_frames, timestamps = extract_frames(video_path, white_threshold, batch_start, batch_end, interval, fps)

        use_limited_templates = False

        for frame_array, timestamp in zip(rgb_frames, timestamps):
            matched_template, match_position = process_frame(
                frame_array, timestamp, template_images, confidence_threshold, white_threshold, csv_output, last_matched_template, 
                last_match_position, search_width, search_height, frame_counter, use_limited_templates
            )

            if matched_template is None:
                use_limited_templates = True
            else:
                last_matched_template = matched_template
                last_match_position = match_position
                use_limited_templates = False

            frame_counter += 1

    processing_done.set()

def main():
    parser = argparse.ArgumentParser(description="Template matching on extracted video frames.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("template_path", type=str, help="Path to template image or directory.")
    parser.add_argument("--interval", type=float, default=5, help="Interval in seconds between extracted frames.")
    parser.add_argument("--output", type=str, default="output", help="Directory to store output CSV (default: output)")
    parser.add_argument("--search_width", type=int, default=100)
    parser.add_argument("--search_height", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--start_time", type=float, help="Start time in seconds (optional)")
    parser.add_argument("--end_time", type=float, help="End time in seconds (optional)")
    args = parser.parse_args()

    matcher_thread = threading.Thread(target=template_matcher, args=(
        args.video_path, args.template_path, args.interval, 0.90, 200,
        args.output, args.search_width, args.search_height, args.batch_size,
        args.start_time, args.end_time
    ))
    matcher_thread.start()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_csv = os.path.join(args.output, f"{video_name}_match_results.csv")
    manual_review_loop(output_csv, args.video_path)
    matcher_thread.join()

if __name__ == "__main__":
    main()
