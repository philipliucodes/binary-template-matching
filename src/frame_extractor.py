import cv2
import os
import argparse

def extract_frame(video_path, time_str, output_dir):
    try:
        if ":" in time_str and "." in time_str:
            minutes, seconds, milliseconds = map(int, time_str.replace(":", ".").split("."))
        elif ":" in time_str:
            minutes, seconds = map(int, time_str.split(":"))
            milliseconds = 0
        else:
            raise ValueError

        total_seconds = minutes * 60 + seconds + milliseconds / 1000
    except ValueError:
        print("Error: Time format must be MM:SS:MS or MM:SS.MS (e.g., '00:44:500' or '00:44.500').")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Could not determine video FPS.")
        cap.release()
        return

    frame_number = round(total_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        print(f"Error: Could not extract a frame at {time_str}.")
        cap.release()
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    formatted_time = time_str.replace(":", "_").replace(".", "_")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}_{formatted_time}.bmp")

    cv2.imwrite(output_path, frame)
    print(f"Frame extracted at {time_str} (frame {frame_number}) and saved to '{output_path}'.")

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Extract a frame from a video at a specific timestamp.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("time", type=str, help="Timestamp in 'MM:SS:MS' or 'MM:SS.MS' format.")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Output directory.")

    args = parser.parse_args()
    extract_frame(args.video_path, args.time, args.output)

if __name__ == "__main__":
    main()
