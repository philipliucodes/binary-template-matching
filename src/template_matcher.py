import cv2
import numpy as np
import argparse
import os
from glob import glob

def load_images(path, grayscale=True):
    """
    Loads an image or all images from a directory.

    :param path: Path to an image file or directory.
    :param grayscale: Whether to load images in grayscale.
    :return: List of loaded images.
    """
    images = []
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    if os.path.isdir(path):
        image_paths = glob(os.path.join(path, "*.[pjP][npP][gG]"))
    elif os.path.isfile(path):
        image_paths = [path]
    else:
        print(f"Error: Invalid path '{path}'. Ensure it exists.")
        return []

    for img_path in image_paths:
        image = cv2.imread(img_path, flag)
        if image is not None:
            images.append((img_path, image))
        else:
            print(f"Warning: Failed to load image '{img_path}'.")

    return images

def binarize_image(image, threshold=200):
    """
    Converts an image to a binary format using a specified threshold.

    :param image: Input grayscale image.
    :param threshold: Threshold value for binarization.
    :return: Binarized image.
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def match_templates(input_images, template_images, threshold=0.9):
    """
    Matches templates within input images using template matching.

    :param input_images: List of (path, image) tuples for input images.
    :param template_images: List of (path, image) tuples for template images.
    :param threshold: Matching threshold.
    :return: Dictionary of matches per input image.
    """
    match_results = {}

    for input_path, input_img in input_images:
        detected_regions = []
        for template_path, template in template_images:
            result = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                w, h = template.shape[::-1]
                detected_regions.append((pt[0], pt[1], w, h, template_path))

        match_results[input_path] = detected_regions

    return match_results

def draw_matches(image, matches):
    """
    Draws rectangles around detected template matches.

    :param image: Input image.
    :param matches: List of detected match coordinates [(x, y, w, h, template_path), ...].
    :return: Image with matches drawn.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for (x, y, w, h, _) in matches:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return output_image

def save_output(image, output_path):
    """
    Saves the processed image with detected template matches.

    :param image: Image with drawn matches.
    :param output_path: Path to save the output image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Output saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Template matching using binarization and direct pixel matching.")
    parser.add_argument("input", type=str, help="Path to the input image or directory.")
    parser.add_argument("template", type=str, help="Path to the template image or directory.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Matching threshold (default: 0.8).")
    parser.add_argument("--white_threshold", type=int, default=200, help="White threshold for binarization (default: 200).")
    parser.add_argument("--output", type=str, default="output/matched_results/", help="Directory to save output images.")

    args = parser.parse_args()

    # Load and process images
    input_images = load_images(args.input)
    template_images = load_images(args.template)

    if not input_images:
        print("Error: No valid input images loaded.")
        return
    if not template_images:
        print("Error: No valid template images loaded.")
        return

    # Binarize images with specified white threshold
    input_images = [(path, binarize_image(img, args.white_threshold)) for path, img in input_images]
    template_images = [(path, binarize_image(img, args.white_threshold)) for path, img in template_images]

    # Perform template matching
    matches = match_templates(input_images, template_images, args.threshold)

    # Draw and save results
    for input_path, detected_regions in matches.items():
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        output_img = draw_matches(image, detected_regions)

        # Generate output path
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{input_name}_matched.png"
        output_path = os.path.join(args.output, output_filename)

        save_output(output_img, output_path)

if __name__ == "__main__":
    main()