from pytoon.animator import animate
from moviepy.editor import VideoFileClip

def add_avatar(audio_file,video_file,output_file,scale=0.5):
    animation = animate(
        audio_file=audio_file
    )

    background_video = VideoFileClip(video_file)
    animation.export(path=output_file, background=background_video, scale=scale)