import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import os
from moviepy.editor import VideoFileClip
import requests
from image_gen.image_gen import image_prompt_similarity_score

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        print("Please ensure Google Chrome is installed on your system.")
        return False

def scrape_links_pexels(driver,topic,orientation='landscape',num_scrolls=5):
    search_url = f"https://www.pexels.com/search/videos/{topic.replace(' ', '%20')}/?orientation={orientation}"
    print(f"Scraping {search_url}")
    driver.get(search_url)
    time.sleep(3)  # Give the page a moment to load initial content and execute JavaScript
    for i in range(num_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for new content to appear after scrolling
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    video_cards = soup.find_all('article', class_=lambda c: c and 'MediaCard_card__6_MG7' in c)
    if not video_cards:
        print("No video cards found with the current selector ('MediaCard_card__6_MG7').")
        return False
    video_results = []
    for i, card in enumerate(video_cards):
        video = card.find('video')
        video_page_url = video['src'] if video else None
        thumbnail = card.find('img')
        thumbnail_url = thumbnail['src'] if thumbnail else None
        if thumbnail_url != 'N/A' and '?' in thumbnail_url:
            thumbnail_url = thumbnail_url.split('?')[0] # Get the base image URL
        download_button = card.find('a', attrs={'download': True})
        direct_download_link = download_button['href'] if download_button else None
        video_results.append({
            'thumbnail_url': thumbnail_url,
            'video_page_url': video_page_url,
            'direct_download_link': direct_download_link
        })
    return video_results

def scrape_links_pixabay(topic,pixaray_api_key):
    url = f"https://pixabay.com/api/videos/?key={pixaray_api_key}={topic.replace(' ','+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = json.loads(str(soup))
    hits = results['hits']
    video_results = []
    for video_card in hits:
        video = video_card['videos']['large']
        video_results.append({
                'thumbnail_url': video['thumbnail'],
                'video_page_url': video_card['pageURL'],
                'direct_download_link': video['url']
            })
    return video_results

def download_file(url, folder_path, filename):
    if not url or url == 'N/A' or "Not found" in url:
        return False
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): # Iterate over content in chunks
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        return False
    except Exception as e:
        return False

def get_main_subject(api_key,paragraph: str) -> str:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    # prompt = f"Given the following sentence, identify and return the main subject in 4 words. If there's an action/verb emphasized give importance to the action also. Do not include any other text.\n\nParagraph: \"{paragraph}\""
    prompt = f"Given the following description, identify and return the main visual subject focusing on the primary events or scenes. Provide a concise phrase, ideally under 8 words. Do not include any other text.\n\nDescription: \"{paragraph}\""
    payload = {
        "contents": [
            {"role": "user",
                "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "text/plain"}}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            main_subject = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return main_subject
        else:
            return "Error: Could not extract main subject from API response."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from Gemini API."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def extract_subject(prompt):
    # prompt = prompt.split('Main Subject/Focus: ')[1]
    # prompt = prompt.split(' Key Characters/Objects:')[0]
    # subject = prompt.strip('.').strip(',').strip()
    subject = prompt
    return subject

def convert_video_aspect_ratio(video_path: str, target_aspect_ratio: str = "16:9"):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None
    try:
        parts = target_aspect_ratio.split(':')
        if len(parts) != 2:
            print(f"Error: Invalid target_aspect_ratio format. Expected 'W:H', got '{target_aspect_ratio}'")
            return None
        target_width_ratio = float(parts[0])
        target_height_ratio = float(parts[1])

        if target_height_ratio == 0:
            print("Error: Target height ratio cannot be zero.")
            return None

        target_ar_value = target_width_ratio / target_height_ratio

        # print(f"Loading video: {video_path}")
        clip = VideoFileClip(video_path,verbose=False)

        original_width = clip.w
        original_height = clip.h
        original_ar_value = original_width / original_height

        new_width = original_width
        new_height = original_height
        if original_ar_value > target_ar_value:
            new_width = int(original_height * target_ar_value)
            # print(f"Video is wider than {target_aspect_ratio}. Cropping width to {new_width}.")
        elif original_ar_value < target_ar_value:
            new_height = int(original_width / target_ar_value)
            # print(f"Video is taller than {target_aspect_ratio}. Cropping height to {new_height}.")
        else:
            # print(f"Video is already at {target_aspect_ratio}. No cropping needed.")
            output_filename = f"{os.path.splitext(video_path)[0]}_{target_aspect_ratio.replace(':','_')}_cropped{os.path.splitext(video_path)[1]}"
            clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            clip.close()
            return output_filename
        x_center = original_width / 2
        y_center = original_height / 2
        cropped_clip = clip.crop(width=new_width, height=new_height, x_center=x_center, y_center=y_center)
        base_name = os.path.splitext(video_path)[0]
        extension = os.path.splitext(video_path)[1]
        output_filename = f"{base_name}_{target_aspect_ratio.replace(':', '_')}_cropped{extension}"
        # print(f"Writing cropped video to: {output_filename}")
        cropped_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        print("Video conversion complete!")
        return output_filename

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return None
    finally:
        if 'clip' in locals() and clip:
            clip.close()
        if 'cropped_clip' in locals() and cropped_clip:
            cropped_clip.close()

def scan_mediafiles(video_results,media_scrape_count,folder_path,score_threshold,prompt_index,aspect_ratio,video_output,source,api_key,prompt):
    video_results = video_results[:media_scrape_count]
    print("Downloading thumbnails...")
    for i in range(len(video_results)):
        download_file(url=video_results[i]['thumbnail_url'],folder_path=f"{folder_path}thumbnails/",filename=f"{i}.jpg")
        approve, score = image_prompt_similarity_score(image_file_path=f"{folder_path}thumbnails/{i}.jpg", target_prompt=prompt,api_key=api_key, score_threshold=score_threshold)
        video_results[i]['score'] = score
    best_media = max(video_results,key=lambda x: x['score'])
    if best_media['score'] > score_threshold:
        print("Downloading video with score:",best_media['score'])
        download_file(url=best_media['direct_download_link'],folder_path=f"{folder_path}prompt_{prompt_index}/",filename="video.mp4")
        print(f"Cropping the video to {aspect_ratio}")
        convert_video_aspect_ratio(video_path=f"{folder_path}prompt_{prompt_index}/video.mp4",target_aspect_ratio=aspect_ratio)
        video_file = download_file(url=best_media['thumbnail_url'],folder_path=f"{folder_path}prompt_{prompt_index}/",filename="thumbnail.jpg")
        video_output[f'{source}_output_file'] = video_file
        video_output[f'{source}_thumbnail'] = f"{folder_path}prompt_{prompt_index}/thumbnail.jpg"
        video_output[f'{source}_score'] = best_media['score']
    else:
        video_output[f'{source}_output_file'] = None
        video_output[f'{source}_thumbnail'] = best_media['thumbnail_url']
        print("The media files are not related to the topic, best score:",best_media['score'])
    return video_output

def scrape_stock_media(data_folder_path,json_file_path,aspect_ratio,score_threshold,api_key,pixaray_api_key):
    json_file = json.load(open(json_file_path))
    image_prompts = json_file["image-prompts"]
    parts = aspect_ratio.split(':')
    orientation = 'portrait'
    if float(parts[0]) > float(parts[1]):
        orientation = 'landscape'
    media_scrape_count = 3
    video_outputs = []
    for prompt_index in image_prompts:
        prompt = image_prompts[prompt_index]
        try:
            subject = extract_subject(prompt)
            subject = get_main_subject(paragraph=subject,api_key=api_key) # using gemini
        except:
            subject = extract_subject(prompt)
            subject = get_main_subject(paragraph=subject,api_key=api_key) # using gemini
        video_output = {
            'prompt_index': prompt_index,
            'aspect_ratio': aspect_ratio,
            'prompt': prompt,
            'subject': subject,
        }
        print(f"Scraping stock media links for prompt #:{prompt_index}/{len(image_prompts)}\nprompt: {prompt}\nsubject: {subject}")
        source = 'pexels'
        print(f"From {source}")
        driver = setup_driver()
        if driver:
            video_results = scrape_links_pexels(driver=driver,topic=subject,orientation=orientation)
            if video_results:
                folder_path=f"{data_folder_path}/stock_media/{source}/"
                video_output = scan_mediafiles(video_results,media_scrape_count,folder_path,score_threshold,prompt_index,aspect_ratio,video_output,source,api_key,prompt)
            else:
                video_output[f'{source}_output_file'] = None
                video_output[f'{source}_thumbnail'] = None
                print('No links found!!!')
            driver.quit()
        else:
            video_output[f'{source}_output_file'] = None
            video_output[f'{source}_thumbnail'] = None
            print("Couldn't setup the driver!!!")

        source = 'pixaray'
        print(f"From {source}")
        video_results = scrape_links_pixabay(topic=subject,pixaray_api_key=pixaray_api_key)
        if video_results:
            folder_path=f"{data_folder_path}/stock_media/{source}/"
            video_output = scan_mediafiles(video_results,media_scrape_count,folder_path,score_threshold,prompt_index,aspect_ratio,video_output,source,api_key,prompt)
        else:
            video_output[f'{source}_output_file'] = None
            video_output[f'{source}_thumbnail'] = None
            print('No links found!!!')
        video_outputs.append(video_output)
    output_path = '/'.join(json_file_path.split('/')[:-1])+'/stock_media.json'
    json.dump(video_outputs,open(output_path,"w"))
    return output_path