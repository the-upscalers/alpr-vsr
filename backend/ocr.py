import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from difflib import SequenceMatcher
from fast_plate_ocr import ONNXPlateRecognizer


class LicensePlateOCR:
    def __init__(self, model_path="global-plates-mobile-vit-v2-model"):
        self.model = ONNXPlateRecognizer(model_path)

    def extract_frames(self, video_path, frame_skip=3):
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray

    def perform_ocr(self, frames):
        ocr_results = []
        for frame in frames:
            processed_frame = self.preprocess_image(frame)
            text = self.model.run(processed_frame)
            if text and len(text) > 0:
                text = text[0].strip().replace(" ", "").upper().replace("_", "")
                if len(text) >= 4:
                    ocr_results.append(text)
        return ocr_results

    def analyze_results(self, ocr_results):
        if not ocr_results:
            return None, 0.0

        # Filter out results with unexpected lengths
        expected_length = Counter(map(len, ocr_results)).most_common(1)[0][0]
        filtered_results = [r for r in ocr_results if len(r) == expected_length]
        if not filtered_results:
            return None, 0.0

        # Cluster similar results
        clusters = []
        for result in filtered_results:
            added = False
            for cluster in clusters:
                if any(
                    SequenceMatcher(None, result, c).ratio() >= 0.8 for c in cluster
                ):
                    cluster.append(result)
                    added = True
                    break
            if not added:
                clusters.append([result])
        largest_cluster = max(clusters, key=len)

        # Calculate character frequencies
        char_frequencies = [{} for _ in range(expected_length)]
        for result in largest_cluster:
            for i, char in enumerate(result):
                char_frequencies[i][char] = char_frequencies[i].get(char, 0) + 1

        # Get the most frequent character for each position
        final_plate = "".join(
            max(freq.items(), key=lambda x: x[1])[0]
            for freq in char_frequencies
            if freq
        )
        confidence = len(largest_cluster) / len(filtered_results)
        return final_plate, confidence

    def get_final_plate(self, video_path):
        frames = self.extract_frames(video_path)
        ocr_results = self.perform_ocr(frames)
        return self.analyze_results(ocr_results)


class LicensePlateOCRVisualizer:
    def __init__(self, ocr_model):
        self.model = ocr_model

    def visualize(self, video_path, frame_skip=3, output_path=None):
        frames = self.model.extract_frames(video_path, frame_skip)
        ocr_results = self.model.perform_ocr(frames)
        final_plate, confidence = self.model.analyze_results(ocr_results)

        if not frames or not ocr_results:
            print("No data to visualize.")
            return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)

        # Create the axes with specific spans
        axes = {}
        axes[0, 0] = fig.add_subplot(gs[0, 0])  # Upscaled Frames
        axes[0, 1] = fig.add_subplot(gs[0, 1])  # Processed Frames
        axes[0, 2] = fig.add_subplot(gs[0, 2])  # Confidence Timeline
        axes[1, 0] = fig.add_subplot(gs[1, 0])  # OCR Timeline
        axes[1, 1] = fig.add_subplot(gs[1, 1])  # Confidence Ranking (spans 2 columns)
        axes[1, 2] = fig.add_subplot(gs[1, 2])  # Character Frequency

        self.plot_frames(axes[0, 0], frames[:6], "Upscaled Frames", gray=False)
        processed_frames = [self.model.preprocess_image(frame) for frame in frames[:6]]
        self.plot_frames(axes[0, 1], processed_frames, "Processed Frames")
        self.plot_confidence_timeline(axes[0, 2], ocr_results)
        self.plot_ocr_timeline(axes[1, 0], ocr_results)
        self.plot_confidence_ranking(axes[1, 1], ocr_results)
        self.plot_char_frequency_heatmap(axes[1, 2], ocr_results)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

        print(f"Final Plate: {final_plate}, Confidence: {confidence:.2f}")

    def plot_frames(self, ax, frames, title, gray=True):
        if not frames:
            return
        rows, cols = 2, 3
        grid = np.zeros(
            (
                rows * frames[0].shape[0],
                cols * frames[0].shape[1],
                3 if not gray else 1,
            ),
            dtype=frames[0].dtype,
        )
        for idx, frame in enumerate(frames[: rows * cols]):
            i, j = divmod(idx, cols)
            if gray and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not gray and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if gray and len(frame.shape) == 2:
                frame = frame[:, :, np.newaxis]
            y1, y2 = i * frame.shape[0], (i + 1) * frame.shape[0]
            x1, x2 = j * frame.shape[1], (j + 1) * frame.shape[1]
            if gray:
                grid[y1:y2, x1:x2] = frame
            else:
                grid[y1:y2, x1:x2, :] = frame
        if gray:
            ax.imshow(grid, cmap="gray")
        else:
            ax.imshow(grid)
        ax.set_title(title)
        ax.axis("off")

    def plot_ocr_timeline(self, ax, ocr_results):
        if not ocr_results:
            return
        result_counts = Counter(ocr_results)
        ax.bar(result_counts.keys(), result_counts.values(), color="skyblue")
        ax.set_title("OCR Results Distribution")
        ax.set_xlabel("Detected Plate Numbers")
        ax.set_ylabel("Frequency")
        plt.setp(ax.get_xticklabels(), rotation=45)

    def plot_char_frequency_heatmap(self, ax, ocr_results):
        if not ocr_results:
            return
        most_common_length = Counter(map(len, ocr_results)).most_common(1)[0][0]
        filtered_results = [r for r in ocr_results if len(r) == most_common_length]
        unique_chars = sorted(set("".join(filtered_results)))
        char_freq = np.zeros((len(unique_chars), most_common_length))
        for result in filtered_results:
            for pos, char in enumerate(result):
                char_freq[unique_chars.index(char)][pos] += 1
        char_freq /= len(filtered_results)
        im = ax.imshow(char_freq, cmap="YlOrRd", aspect="auto")
        ax.set_title("Character Frequency by Position")
        ax.set_yticks(range(len(unique_chars)))
        ax.set_yticklabels(unique_chars)
        ax.set_xticks(range(most_common_length))
        ax.set_xlabel("Character Position")
        plt.colorbar(im, ax=ax)

    def plot_confidence_timeline(self, ax, ocr_results):
        if not ocr_results:
            return
        # Calculate similarity scores between consecutive readings
        similarities = []
        for i in range(len(ocr_results) - 1):
            similarity = SequenceMatcher(
                None, ocr_results[i], ocr_results[i + 1]
            ).ratio()
            similarities.append(similarity)

        ax.plot(similarities, color="lime", linewidth=2)
        ax.fill_between(range(len(similarities)), similarities, alpha=0.3, color="lime")
        ax.set_title("Reading Confidence Timeline")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Confidence Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def plot_confidence_ranking(self, ax, ocr_results):
        if not ocr_results:
            return

        # Calculate confidence scores
        readings = Counter(ocr_results)
        total_readings = len(ocr_results)
        confidence_ranking = [
            (plate, count / total_readings) for plate, count in readings.items()
        ]

        # Sort and get top 10
        top_10 = sorted(confidence_ranking, key=lambda x: x[1], reverse=True)[:10]

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Create title with nice background (fixed bbox parameters)
        ax.text(
            0.5,
            0.95,
            "Top License Plate Readings",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="navy", alpha=0.3, edgecolor="none", pad=8),
        )

        # Create headers
        ax.text(0.1, 0.85, "RANK", fontweight="bold", fontsize=12, color="gray")
        ax.text(0.25, 0.85, "PLATE", fontweight="bold", fontsize=12, color="gray")
        ax.text(0.6, 0.85, "CONFIDENCE", fontweight="bold", fontsize=12, color="gray")

        # Add a subtle line under headers
        ax.axhline(y=0.82, xmin=0.08, xmax=0.92, color="gray", alpha=0.3, linewidth=1)

        # Add entries
        for i, (plate, confidence) in enumerate(top_10):
            y_pos = 0.75 - (i * 0.07)

            rank_text = f"#{i+1}"
            ax.text(0.1, y_pos, rank_text, ha="left", va="center", fontsize=11)
            ax.text(
                0.25,
                y_pos,
                plate,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                0.6, y_pos, f"{confidence:.1%}", ha="left", va="center", fontsize=11
            )

            # Add subtle alternating background for better readability
            if i % 2 == 0:
                ax.axhspan(
                    y_pos - 0.03,
                    y_pos + 0.03,
                    xmin=0.08,
                    xmax=0.92,
                    color="white",
                    alpha=0.05,
                )


def perform_ocr_on_video(ocr_model, video_file, output_path):
    visualizer = LicensePlateOCRVisualizer(ocr_model)
    visualizer.visualize(video_file, output_path=output_path)


def main():
    video_file = "temp/upscaled/output_video_2.mp4"
    ocr = LicensePlateOCR()
    visualizer = LicensePlateOCRVisualizer(ocr)
    visualizer.visualize(video_file, output_path="temp/ocr_viz.png")


if __name__ == "__main__":
    main()
