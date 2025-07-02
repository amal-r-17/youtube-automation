import base64
import requests
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips, vfx
from tqdm import tqdm
import json

def video_diffusion(input_file,output_file,api_key):
    with open(input_file, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    payload = {
        "image": f"data:image/jpeg;base64,{base64_image}",
        "cfg_scale": 9,
        "seed": 35,
        "motion_bucket_id": 127   #can't change
    }
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    response_body = response.json()
    try:
        if response_body['finish_reason'] == 'SUCCESS':
            video_bytes = base64.b64decode(response_body['video'])
            with open(output_file, "wb") as f:
                f.write(video_bytes)

            return f"Video saved to {output_file}"
    except:
        try:
            return f"Error: {response_body['detail']}"
        except:
            return "Error"

def img_to_video(IMAGE_FOLDER,VIDEO_FOLDER,api_key):
    for img in tqdm(os.listdir(IMAGE_FOLDER)):
        img_file = os.path.join(IMAGE_FOLDER, img)
        img_num = img.split('.')[0]
        output_file = os.path.join(VIDEO_FOLDER, f"{img_num}.mp4")
        result = video_diffusion(img_file, output_file, api_key)
        # print(img,'-->',result)
        # break

def process_video(
    input_filepath,
    output_filepath,
    slow_motion_factor=0.5, # 0.5 means half speed (twice as long), 2.0 means double speed (half as long)
    trim_start_time=0,     # Start time in seconds
    trim_duration=10,      # Desired duration of the output video in seconds
):
    try:
        clip = VideoFileClip(input_filepath)
        # slow_clip = clip.fx(vfx.speedx, slow_motion_factor)
        slow_clip = clip
        effective_trim_end = min(trim_start_time + trim_duration, slow_clip.duration)
        trimmed_clip = slow_clip.subclip(trim_start_time, effective_trim_end)
        actual_trimmed_duration = trimmed_clip.duration
        final_clip = trimmed_clip
        if actual_trimmed_duration < trim_duration:
            num_loops = int(trim_duration // actual_trimmed_duration)
            remaining_duration = trim_duration % actual_trimmed_duration
            clips_to_concatenate = [trimmed_clip] * num_loops
            if remaining_duration > 0:
                clips_to_concatenate.append(trimmed_clip.subclip(0, remaining_duration))
            final_clip = concatenate_videoclips(clips_to_concatenate)

        final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac",logger=None,verbose=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'clip' in locals() and clip:
            clip.close()
        if 'slow_clip' in locals() and slow_clip:
            slow_clip.close()
        if 'trimmed_clip' in locals() and trimmed_clip:
            trimmed_clip.close()
        if 'final_clip' in locals() and final_clip and final_clip != trimmed_clip:
            final_clip.close()

def diffuse_images(IMAGE_FOLDER,api_key):
    img_timings = '/'.join(IMAGE_FOLDER.split("/")[:-1]) + f'/audio/img_timings.json'
    VIDEO_FOLDER = '/'.join(IMAGE_FOLDER.split("/")[:-1]) + '/videos'
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    img_to_video(IMAGE_FOLDER, VIDEO_FOLDER, api_key)
    img_timings = json.load(open(img_timings))
    print('Adjusting clip durations...')
    for img in tqdm(img_timings):
        img_path = img['image_path']
        img_filename = img_path.split("/")[-1][:-4]
        duration = img['image_duration']
        video_inp = VIDEO_FOLDER + f'/{img_filename}.mp4'
        video_op = VIDEO_FOLDER + f'/processed_{img_filename}.mp4'
        if os.path.exists(video_inp):
            process_video(input_filepath=video_inp, output_filepath=video_op, trim_duration=duration)
    return VIDEO_FOLDER

