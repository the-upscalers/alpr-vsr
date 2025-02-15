import cv2
import numpy as np
from collections import Counter
import re
from difflib import SequenceMatcher
from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")


def extract_frames(video_path, frame_skip=3):
    """Extracts frames from a video, skipping some for efficiency."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames


def preprocess_image(image):
    """Preprocesses image for better OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(
        gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )  # Apply thresholding
    return thresh


def perform_tesseract_ocr(frames):
    """Runs Tesseract OCR on each frame and collects results."""
    ocr_results = []
    for frame in frames:
        processed_frame = preprocess_image(frame)

        text = m.run(processed_frame)

        print("OCR Result:", text)

        # Clean up text output
        text = text[0].strip().replace(" ", "").upper()

        if len(text) > 3:  # Ignore very short/noisy results
            ocr_results.append(text)

    return ocr_results


def filter_plate_format(ocr_results):
    """
    Filters out unlikely plate sequences using regex.
    Example: Only allow formats like "ABC123", "AB 1234", "ABC-123".
    """
    plate_pattern = re.compile(
        r"^[A-Z]{1,3} ?[0-9]{2,4}$"
    )  # Adjust for your country's plate format

    filtered_results = [text for text in ocr_results if plate_pattern.match(text)]

    return (
        filtered_results if filtered_results else ocr_results
    )  # Fallback to original if none match


def majority_voting(ocr_results):
    """Uses majority voting to determine the most likely license plate."""
    if not ocr_results:
        return ""

    common_results = Counter(ocr_results).most_common()
    best_guess = common_results[0][0]  # Take the most frequent result

    return best_guess


def refine_with_similarity(ocr_results):
    """Uses Levenshtein similarity to stabilize the most probable license plate."""
    if not ocr_results:
        return ""

    base_plate = ocr_results[0]  # Take the first detected plate as reference
    for text in ocr_results[1:]:
        if (
            SequenceMatcher(None, base_plate, text).ratio() > 0.7
        ):  # Similarity threshold
            base_plate = text  # Choose the closest match

    return base_plate


def correct_ocr_errors(text):
    """Fixes common OCR misreads (O->0, S->5, I->1, etc.)."""
    replacements = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    return "".join(replacements.get(c, c) for c in text)


def get_final_plate(video_path):
    """
    Main function to extract and refine license plate text from a video.
    """
    frames = extract_frames(video_path)
    ocr_results = perform_tesseract_ocr(frames)

    print("OCR Results:", ocr_results)

    # Filter results to match license plate formats
    filtered_results = filter_plate_format(ocr_results)

    # Align and vote on the best sequence
    best_guess = majority_voting(filtered_results)

    # Apply similarity refinement
    refined_plate = refine_with_similarity(filtered_results)

    # Apply final post-processing
    final_plate = correct_ocr_errors(refined_plate)

    return final_plate


# Example Usage:
video_file = "temp/output_video_1.mp4"
final_license_plate = get_final_plate(video_file)
print("Final Detected License Plate:", final_license_plate)
