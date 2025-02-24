import os
import argparse
import numpy as np
from PIL import Image
import cv2

def is_image(file_path):
    """Checks if the file is an image based on its extension."""
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def get_images(path):
    """Returns a list of image paths from a given file or directory."""
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path) if is_image(f)]
    elif os.path.isfile(path) and is_image(path):
        return [path]
    else:
        return []

def transform_pixels(image_array, alpha_channel, white_threshold):
    """Transforms an image to a binary mask based on transparency and whiteness."""
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

def template_matcher(input_path, template_path, confidence_threshold, white_threshold, output_dir):
    """Performs template matching on input images using the provided template(s)."""
    os.makedirs(output_dir, exist_ok=True)
    
    input_images = get_images(input_path)
    template_images = get_images(template_path)
    
    if not input_images:
        print(f"Error: No valid input images found in '{input_path}'")
        return
    if not template_images:
        print(f"Error: No valid template images found in '{template_path}'")
        return
    
    for input_filename in input_images:
        input_image = Image.open(input_filename).convert("RGBA")
        input_array = np.array(input_image)
        input_alpha = input_array[:, :, 3]
        input_transformed = transform_pixels(input_array, input_alpha, white_threshold)
        
        best_template = None
        max_match_score = 0
        best_match_positions = []
        best_template_size = (0, 0)
        best_match_count = 0
        best_match_percentage = 0.0
        
        for template_filename in template_images:
            template_image = Image.open(template_filename).convert("RGBA")
            template_array = np.array(template_image)
            template_alpha = template_array[:, :, 3]
            template_transformed = transform_pixels(template_array, template_alpha, white_threshold)
            
            ih, iw = input_transformed.shape
            th, tw = template_transformed.shape
            
            matches = []
            match_score_total = 0
            total_comparisons = 0
            for y in range(ih - th + 1):
                for x in range(iw - tw + 1):
                    roi = input_transformed[y:y+th, x:x+tw]
                    mask = template_alpha > 0
                    total_pixels = np.count_nonzero(mask)
                    
                    if total_pixels > 0:
                        matching_pixels = np.sum(roi[mask] == template_transformed[mask])
                        match_score = matching_pixels / total_pixels
                        
                        if match_score >= confidence_threshold:
                            matches.append((x, y))
                            match_score_total += match_score
                            total_comparisons += 1
            
            if total_comparisons > 0:
                average_match_score = (match_score_total / total_comparisons) * 100
            else:
                average_match_score = 0.0
            
            if match_score_total > max_match_score:
                max_match_score = match_score_total
                best_template = os.path.basename(template_filename)
                best_match_positions = matches
                best_template_size = (tw, th)
                best_match_count = len(matches)
                best_match_percentage = average_match_score
        
        if best_template and best_match_count > 0:
            result_array = np.array(input_image)
            for (x, y) in best_match_positions:
                cv2.rectangle(result_array, (x, y), (x + best_template_size[0], y + best_template_size[1]), (255, 0, 0, 255), 2)
            
            output_image = Image.fromarray(result_array)
            output_name = os.path.splitext(os.path.basename(input_filename))[0] + "_matched.png"
            output_path = os.path.join(output_dir, output_name)
            output_image.save(output_path)
            
            print(f"Input '{os.path.basename(input_filename)}': Best template '{best_template}' with {best_match_count} matches (Average Match: {best_match_percentage:.2f}%).")
        else:
            print(f"Input '{os.path.basename(input_filename)}': No template matches found.")

def main():
    """Parses command-line arguments and runs the template matcher."""
    parser = argparse.ArgumentParser(description="Template matching using image masks.")
    parser.add_argument("input_path", type=str, help="Path to input image or directory.")
    parser.add_argument("template_path", type=str, help="Path to template image or directory.")
    parser.add_argument("--confidence_threshold", type=float, default=0.90, help="Threshold for matching confidence (default: 0.90)")
    parser.add_argument("--white_threshold", type=int, default=200, help="Threshold for defining near-white pixels (default: 200)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save matched images (default: output/)")
    
    args = parser.parse_args()
    
    template_matcher(args.input_path, args.template_path, args.confidence_threshold, args.white_threshold, args.output)

if __name__ == "__main__":
    main()
