import os
import random
from moviepy.editor import *
from PIL import Image
import numpy as np
import json

def resize_image_to_target(image_path, target_size=(1920, 1080)):
    """
    Resize image to target size while maintaining aspect ratio by adding black bars
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scaling to fit within target size
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider - fit to width
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        else:
            # Image is taller - fit to height
            new_height = target_size[1]
            new_width = int(target_size[1] * img_ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and black background
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Paste resized image in center
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return np.array(new_img)

def create_zoom_effect(image_array, duration, zoom_type='in', zoom_factor=1.3):
    """
    Create zoom in/out effect for an image
    """
    def make_frame(t):
        # Calculate zoom progress (0 to 1)
        progress = t / duration
        
        if zoom_type == 'in':
            current_zoom = 1 + (zoom_factor - 1) * progress
        else:  # zoom out
            current_zoom = zoom_factor - (zoom_factor - 1) * progress
        
        h, w = image_array.shape[:2]
        
        # Calculate crop dimensions
        crop_h = int(h / current_zoom)
        crop_w = int(w / current_zoom)
        
        # Calculate crop position (center)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        # Crop and resize back to original size
        cropped = image_array[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Use PIL for high-quality resizing
        pil_img = Image.fromarray(cropped)
        resized = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        
        return np.array(resized)
    
    return VideoClip(make_frame, duration=duration)

def create_video_from_images(img_timing_file,images_folder, audio_file, output_file, target_size=(1920, 1080),
                             video_diffusion='False'):

    imges_len = len(os.listdir(images_folder))
    image_timings = json.load(open(img_timing_file))

    print(f"Found {len(image_timings)} images")
    
    # Load audio to get duration
    audio_clip = AudioFileClip(audio_file)
    total_duration = audio_clip.duration
    
    # Calculate timing
    transition_duration = 0.5  # 0.5 seconds for transitions

    print(f"Transition duration: {transition_duration} seconds")
    
    # Create video clips
    clips = []
    zoom_effects = ['in', 'out', 'none']  # Different effects to cycle through
    for i, image_file in enumerate(image_timings):
        image_path = image_file['image_path']
        img_filename = image_path.split("/")[-1][:-4]
        image_display_time = image_file['image_duration'] - transition_duration
        diffused_video_path = '/'.join(image_path.split("/")[:-2]) + f'/videos/processed_{img_filename}.mp4'
        if video_diffusion == 'False' or os.path.exists(diffused_video_path) == False:
            resized_image = resize_image_to_target(image_path, target_size)

            # Choose zoom effect (cycle through different effects)
            zoom_type = zoom_effects[i % len(zoom_effects)]

            if zoom_type == 'none':
                # Static image
                clip = ImageClip(resized_image, duration=image_display_time)
            else:
                # Apply zoom effect
                clip = create_zoom_effect(resized_image, image_display_time, zoom_type)
                # Add transition (except for the last image)
            clips.append(clip)
            if i < len(image_timings) - 1:
                # Create crossfade transition
                next_image_path = image_timings[i + 1]['image_path']
                next_resized_image = resize_image_to_target(next_image_path, target_size)

                # Create a short clip for transition
                transition_clip = ImageClip(next_resized_image, duration=transition_duration)

                # Apply crossfade effect
                clips[-1] = clips[-1].crossfadeout(transition_duration / 2)
                transition_clip = transition_clip.crossfadein(transition_duration / 2)

                clips.append(transition_clip)
        else:
            clip = VideoFileClip(diffused_video_path)
            clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    
    # Trim video to match audio duration exactly
    final_video = final_video.subclip(0, total_duration)
    
    # Add audio
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(
        output_file,
        fps=24,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        verbose=False,
        logger=None
    )
    
    print("Video creation completed!")
    
    # Clean up
    audio_clip.close()
    final_video.close()
