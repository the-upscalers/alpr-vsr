import cv2
from fast_plate_ocr import ONNXPlateRecognizer
from collections import Counter
from difflib import SequenceMatcher
import numpy as np


class LicensePlateOCR:
    def __init__(self, model_path="global-plates-mobile-vit-v2-model"):
        self.model = ONNXPlateRecognizer(model_path)

    def extract_frames(self, video_path, frame_skip=3):
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

    def preprocess_image(self, image):
        """Preprocesses image for better OCR accuracy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(
            gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )  # Apply thresholding
        return thresh

    def perform_ocr(self, frames):
        """Runs OCR on each frame and collects results."""
        ocr_results = []

        for frame in frames:
            processed_frame = self.preprocess_image(frame)
            text = self.model.run(processed_frame)

            # Clean up text output
            if text and len(text) > 0:
                text = text[0].strip().replace(" ", "").upper()
                text = text.replace("_", "")

                # Basic validation
                if len(text) >= 4:  # Minimum realistic plate length
                    ocr_results.append(text)

        return ocr_results

    def calculate_string_similarity(self, str1, str2):
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, str1, str2).ratio()

    def find_most_common_length(self, results):
        """Find the most common length among OCR results."""
        lengths = [len(result) for result in results]
        return Counter(lengths).most_common(1)[0][0]

    def cluster_similar_results(self, results, similarity_threshold=0.8):
        """Group similar OCR results together."""
        clusters = []

        for result in results:
            added_to_cluster = False

            for cluster in clusters:
                if any(
                    self.calculate_string_similarity(result, existing)
                    >= similarity_threshold
                    for existing in cluster
                ):
                    cluster.append(result)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([result])

        return clusters

    def get_character_frequencies(self, results, expected_length):
        """Analyze character frequencies at each position."""
        char_frequencies = [{} for _ in range(expected_length)]

        for result in results:
            if len(result) == expected_length:
                for i, char in enumerate(result):
                    char_frequencies[i][char] = char_frequencies[i].get(char, 0) + 1

        return char_frequencies

    def construct_final_plate(self, char_frequencies):
        """Construct final plate number using most frequent characters."""
        final_plate = ""

        for pos_freq in char_frequencies:
            if pos_freq:
                most_common_char = max(pos_freq.items(), key=lambda x: x[1])[0]
                final_plate += most_common_char

        return final_plate

    def get_final_plate(self, video_path, min_confidence=0.6):
        """
        Main function to extract and refine license plate text from a video.
        Returns tuple of (plate_number, confidence_score)
        """
        # Extract and process frames
        frames = self.extract_frames(video_path)
        ocr_results = self.perform_ocr(frames)

        if not ocr_results:
            return None, 0.0

        # Find most common length to filter out obvious errors
        expected_length = self.find_most_common_length(ocr_results)
        filtered_results = [r for r in ocr_results if len(r) == expected_length]

        if not filtered_results:
            return None, 0.0

        # Cluster similar results
        clusters = self.cluster_similar_results(filtered_results)

        print(clusters)

        # Get the largest cluster
        largest_cluster = max(clusters, key=len)

        # Calculate character frequencies for the largest cluster
        char_frequencies = self.get_character_frequencies(
            largest_cluster, expected_length
        )

        # Construct final plate number
        final_plate = self.construct_final_plate(char_frequencies)

        # Calculate confidence score
        cluster_size = len(largest_cluster)
        total_results = len(filtered_results)
        confidence_score = cluster_size / total_results if total_results > 0 else 0.0

        if confidence_score < min_confidence:
            return None, confidence_score

        return final_plate, confidence_score


# Example usage:
def main():
    video_file = "temp/output_video_1.mp4"
    ocr = LicensePlateOCR()
    final_plate, confidence = ocr.get_final_plate(video_file)

    if final_plate:
        print(f"Detected License Plate: {final_plate} (Confidence: {confidence:.2f})")
    else:
        print("Could not determine license plate with sufficient confidence")


if __name__ == "__main__":
    main()
