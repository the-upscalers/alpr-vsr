import cv2
import sys

def trim_video(input_path, output_path, num_frames):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Trimmed video saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python trim_video.py <input_video> <output_video> <num_frames>")
    else:
        input_video = sys.argv[1]
        output_video = sys.argv[2]
        num_frames = int(sys.argv[3])
        trim_video(input_video, output_video, num_frames)


# python trim-video.py "/mnt/c/Users/Dan/Projects/alpr-vsr/frontend/videos/street.mp4" "/mnt/c/Users/Dan/Projects/alpr-vsr/frontend/videos/street-trim-uhd.mp4" 600