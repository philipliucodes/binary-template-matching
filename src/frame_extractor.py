import cv2
import os
import argparse

def extract_frame(video_path, time_str, output_dir):
    """
    Extracts a frame from the video at the specified time (MM:SS:MS or MM:SS.MS) 
    and saves it to the specified output directory.

    :param video_path: Path to the input video file.
    :param time_str: Time in "MM:SS:MS" or "MM:SS.MS" format.
    :param output_dir: Directory where the extracted frame will be saved.
    """
    # Convert MM:SS:MS or MM:SS.MS to total milliseconds
    try:
        if ":" in time_str and "." in time_str:
            minutes, seconds, milliseconds = map(int, time_str.replace(":", ".").split("."))
        elif ":" in time_str:
            minutes, seconds = map(int, time_str.split(":"))
            milliseconds = 0
        else:
            raise ValueError

        time_milliseconds = (minutes * 60 + seconds) * 1000 + milliseconds
    except ValueError:
        print("Error: Time format must be MM:SS:MS or MM:SS.MS (e.g., '00:44:500' or '00:44.500').")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Set position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, time_milliseconds)

    # Read the frame
    success, frame = cap.read()
    if not success:
        print(f"Error: Could not extract a frame at {time_str}. Ensure the timestamp is within the video duration.")
        cap.release()
        return

    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Format timestamp for filename (avoid special characters)
    formatted_time = time_str.replace(":", "_").replace(".", "_")

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
    parser = argparse.ArgumentParser(description="Extract a frame from a video at a specific timestamp (MM:SS:MS or MM:SS.MS).")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("time", type=str, help="Timestamp in 'MM:SS:MS' or 'MM:SS.MS' format (e.g., '00:44:500' or '00:44.500').")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Output directory (default: 'extracted_frames').")

    args = parser.parse_args()
    extract_frame(args.video_path, args.time, args.output)

if __name__ == "__main__":
    main()
