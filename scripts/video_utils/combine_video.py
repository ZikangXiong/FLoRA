import os
from moviepy.editor import VideoFileClip, CompositeVideoClip
from PIL import Image
import numpy as np


def resize_clip(clip, newsize):
    def resize_frame(gf, ts):
        frame = Image.fromarray(gf(ts))
        frame = frame.resize(newsize, Image.LANCZOS)  # type: ignore
        return np.array(frame)

    return clip.fl(resize_frame)


def get_video_files_recursively(folder):
    video_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(os.path.join(root, file))
    return video_files


def create_video_gallery(input_folder, output_file, grid_size=(10, 6), duration=None):
    # Get all video files from the input folder and its subdirectories
    video_files = get_video_files_recursively(input_folder)

    if not video_files:
        print("No video files found in the input folder or its subdirectories.")
        return

    # Load video clips
    clips = [VideoFileClip(f) for f in video_files]

    # Calculate the size for each video in the grid
    screen_w, screen_h = 1920, 1080  # Assuming a 1080p output
    clip_w, clip_h = screen_w // grid_size[0], screen_h // grid_size[1]

    # Resize and position clips
    positioned_clips = []
    for i, clip in enumerate(
        clips[: grid_size[0] * grid_size[1]]
    ):  # Limit to grid size
        row = i // grid_size[0]
        col = i % grid_size[0]
        resized_clip = resize_clip(clip, (clip_w, clip_h)).set_position(
            (col * clip_w, row * clip_h)
        )
        positioned_clips.append(resized_clip)

    # Create the composite video
    final_clip = CompositeVideoClip(positioned_clips, size=(screen_w, screen_h))

    # Set duration if specified, otherwise use the shortest clip's duration
    if duration:
        final_clip = final_clip.subclip(0, duration)
    else:
        final_clip = final_clip.subclip(0, min(c.duration for c in clips))

    # Write output video
    final_clip.write_videofile(output_file, codec="libx264")

    # Close clips
    for clip in clips:
        clip.close()


# Example usage with the specified input folder
input_folder = "docs/.vuepress/public/scenarios/videos"
output_file = "output_gallery_10x6.mp4"
create_video_gallery(
    input_folder, output_file, grid_size=(10, 6), duration=20
)  # Creates a 10x6 grid, 60 seconds long
