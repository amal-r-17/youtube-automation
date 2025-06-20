import base64
import requests
import os

def video_diffusion(input_file,output_file,api_key):
    with open(input_file, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    payload = {
        "image": f"data:image/jpeg;base64,{base64_image}",
        "cfg_scale": 3.1,
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
    for img in os.listdir(IMAGE_FOLDER):
        img_file = os.path.join(IMAGE_FOLDER, img)
        img_num = img.split('.')[0]
        output_file = os.path.join(VIDEO_FOLDER, f"{img_num}.mp4")
        result = video_diffusion(img_file, output_file, api_key)
        print(img,'-->',result)
        # break

IMAGE_FOLDER = "data/BTS_IS_BACK!_What_You_MISSED!/images"
VIDEO_FOLDER = "data/BTS_IS_BACK!_What_You_MISSED!/videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
api_key = "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"
img_to_video(IMAGE_FOLDER,VIDEO_FOLDER,api_key)
