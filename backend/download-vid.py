import os
import yt_dlp
from typing import List

def download_videos(urls: List[str], output_dir: str = "downloads") -> None:
    """
    Downloads videos from YouTube using yt-dlp at the highest available quality.

    Args:
        urls (List[str]): List of YouTube video URLs.
        output_dir (str): Directory to save the downloaded videos. Defaults to "downloads".
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # yt-dlp options for highest quality download
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Best quality MP4
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Output template
        'merge_output_format': 'mp4',  # Merge into MP4
        'quiet': False,  # Show progress
        'no_warnings': False,  # Show warnings if any
        'progress_hooks': [lambda d: print_progress(d)],  # Progress hook
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"Downloading: {url}")
                ydl.download([url])
                print(f"Finished downloading: {url}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")

def print_progress(d: dict) -> None:
    """Prints download progress."""
    if d['status'] == 'downloading':
        print(f"Downloading... {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")

def read_urls_from_file(file_path: str) -> List[str]:
    """
    Reads URLs from a text file (one URL per line).

    Args:
        file_path (str): Path to the text file containing URLs.

    Returns:
        List[str]: List of URLs.
    """
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

if __name__ == "__main__":
    # Path to the file containing YouTube URLs (one per line)
    input_file = "youtube_urls.txt"

    # Directory to save downloads (default: "downloads")
    output_directory = "downloads"

    # Read URLs from file
    try:
        video_urls = read_urls_from_file(input_file)
        if not video_urls:
            print(f"No URLs found in {input_file}.")
        else:
            print(f"Found {len(video_urls)} URLs. Starting download...")
            download_videos(video_urls, output_directory)
            print("All downloads completed!")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")