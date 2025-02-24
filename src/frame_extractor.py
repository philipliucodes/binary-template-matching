import cv2
import os
import argparse

def extract_frame(video_path, time_str, output_dir):
    """
    Extracts a frame from the video at the specified time (MM:SS) and saves it to the specified output directory.

    :param video_path: Path to the input video file.
    :param time_str: Time in "MM:SS" format.
    :param output_dir: Directory where the extracted frame will be saved.
    """
    # Convert MM:SS to total seconds
    try:
        minutes, seconds = map(int, time_str.split(":"))
        time_seconds = minutes * 60 + seconds
    except ValueError:
        print("Error: Time format must be MM:SS (e.g., '00:44').")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Set position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)

    # Read the frame
    success, frame = cap.read()
    if not success:
        print(f"Error: Could not extract a frame at {time_str}. Ensure the timestamp is within the video duration.")
        cap.release()
        return

    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Format timestamp for filename (avoid special characters)
    formatted_time = time_str.replace(":", "_")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file path
    output_filename = f"{video_name}_{formatted_time}.bmp"
    output_path = os.path.join(output_dir, output_filename)

    # Save the frame as BMP (lossless)
    cv2.imwrite(output_path, frame)
    print(f"Frame extracted at {time_str} and saved to '{output_path}'.")

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Extract a frame from a video at a specific timestamp.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("time", type=str, help="Timestamp in 'MM:SS' format (e.g., '00:44').")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Output directory (default: 'extracted_frames').")

    args = parser.parse_args()
    extract_frame(args.video_path, args.time, args.output)

if __name__ == "__main__":
    main()