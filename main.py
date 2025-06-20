import os, re
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import time
import datetime
load_dotenv()  # take environment variables
from script.contentbuilder import ContentCreator
from image_gen.image_gen import generate_images,generate_images_flux  # Gen images
from audio_gen.audio_generator import text_to_speech_file
from audio_gen.audio_generator_gemini import generate_audio
from video_gen.video_generator import create_video_from_images
from video_gen.caption_generator import generate_captions
from video_gen.avatar import add_avatar
from video_gen.audio_video_sync import fetch_audio_timings,calculate_media_timings
from context_extract.Scrapper import ContentSummarizer
from stock_media.stock_media_scrapper import scrape_stock_media
from context_extract.topic_grounding import grounding_gemini

def remove_enclosed_segments(text):
    return re.sub(r"\*\*\([^)]*\)\*\*", "", text)

def calc_time(start):
    end = time.time()
    time_taken_seconds = end - start
    time_delta = datetime.timedelta(seconds=time_taken_seconds)
    minutes = int(time_delta.total_seconds() // 60)
    seconds = int(time_delta.total_seconds() % 60)
    formatted_time = f"{minutes:02d}:{seconds:02d}"
    return formatted_time

def main():
    load_dotenv()
    """
    Main function to demonstrate the content creation workflow.
    """
    # Get configuration from environment variables or use defaults
    llm_provider = os.getenv("LLM_TYPE", "gemini")

    # Handle API key or base URL based on provider
    api_key = None
    if llm_provider == "ollama":
        api_key = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        api_key_name = f"{llm_provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_name)

    # Get model name if specified
    model_name = os.environ.get(f"{llm_provider.upper()}_MODEL")

    # ELEVEN LABS API KEY
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    # IMAGE GEN API KEY
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    PIXARAY_API_KEY = os.getenv("PIXARAY_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    TOGETHER_AI_API_KEY = os.environ["TOGETHER_AI_API_KEY"]

    SAVE_DIR = os.getenv("SAVE_DIR")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Video size
    # TARGET_SIZE = (1280, 720) #(1920, 1080)

    # VIDEO PARAMS
    MAX_LENGTH_MINS = float(os.getenv("MAX_LENGTH_MINS", 1))
    NUM_IMAGE_PROMPTS = int(os.getenv("NUM_IMAGE_PROMPTS", 10))
    NUM_IMAGES_PER_PROMPTS = int(os.getenv("NUM_IMAGES_PER_PROMPTS" ,2))
    TARGET_SIZE = [int(val) for val in os.getenv("VIDEO_DIMENSION", "1280,720").split(",")]
    ASPECT_RATIO = os.getenv("ASPECT_RATIO" ,"16:9")

    # IMAGE GEN
    IMG_SIMILARITY_THRESHOLD = float(os.getenv("IMG_SIMILARITY_THRESHOLD", 0.7))

    # AUDIO 
    GENERATE_CAPTIONS = os.getenv("GENERATE_CAPTIONS", True)
    stock_media = os.getenv("INCLUDE_STOCK_MEDIA", True)
    GENERATE_SHORTS = os.getenv("GENERATE_SHORTS", True)
    ADD_AVATAR = os.getenv("INCLUDE_AVATAR", True)
    AUDIO_PACE =  float(os.getenv("AUDIO_PACE", 1.1))
    TTS_MODEL = os.getenv("TTS_MODEL")
    IMG_MODEL = os.getenv("IMG_MODEL")
    CONTENT_SCRAPER_MODEL = os.getenv("CONTENT_SCRAPER_MODEL")

    # Instantiate the content creator
    creator = ContentCreator(
        llm_provider=llm_provider,
        api_key=api_key,
        model_name=model_name,
        MAX_VIDEO_LENGTH=MAX_LENGTH_MINS, NUM_IMG_PROMPTS=NUM_IMAGE_PROMPTS, IMAGE_DIM=TARGET_SIZE, ASPECT_RATIO=ASPECT_RATIO
    )

    # Get topic from command line arguments or use a default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "what if NASA had the budget of US military?"
    search_web = bool(sys.argv[2]) if len(sys.argv) > 2 else False

    print(f"Generating YouTube content for topic: '{topic}'")
    print(f"Using LLM provider: {llm_provider}")

    # Getting the context for the script generation by scrapping
    if search_web:
        start = time.time()
        print(f'Scrapping {topic} content',end=' ')
        if CONTENT_SCRAPER_MODEL == 'GEMINI':
            print('with Gemini')
            context = grounding_gemini(api_key=GOOGLE_API_KEY, topic=topic)
        else:
            print('with DuckDuckGo')
            scrapper = ContentSummarizer(min_len=20, api_key=GOOGLE_API_KEY)
            context = scrapper.run(topic=topic)
        time_taken = calc_time(start)
        print('Scraping content completed in', time_taken)
    else:
        print('Scraping content disabled')
        context = ""

    # Generate content
    print("Generating script ...")
    start = time.time()
    result = creator.create_content(topic, context=context,height=TARGET_SIZE[1],width=TARGET_SIZE[0],aspect_ratio=ASPECT_RATIO)
    time_taken = calc_time(start)
    print('Generating script completed in', time_taken)
    # Adding context to result
    result["context"] = context
    
    # print("result: ", result)

    title = "_".join(result["meta_data"]["title_suggestions"][0].split())

    VIDEO_SAVE_DIR = os.path.join(SAVE_DIR, title)
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

    # MAKING THE IMAGES AND AUDIO DIR
    IMAGE_DIR = os.path.join(VIDEO_SAVE_DIR, "images")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Audio directory
    AUDIO_DIR = os.path.join(VIDEO_SAVE_DIR, "audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Print the result
    # print("\n=== Generated Script ===")
    # print(result["script"])
    #
    # print("\n=== Generated Image Prompts ===")
    # for section_num, prompt in result["image-prompts"].items():
    #     print(f"Section {section_num}:")
    #     print(prompt)

    # Save to file
    filename = os.path.join(VIDEO_SAVE_DIR, f"youtube_content_{topic.replace(' ', '_')[:20]}.json")
    file_path = creator.save_to_json(result, filename)
    print(f"\nContent saved to: {file_path}\n")

    print("[INFO] Image Generation Started -",IMG_MODEL)
    start = time.time()
    # Gen images
    if IMG_MODEL == 'GEMINI':
        img_metadata_file = generate_images(result, IMAGE_DIR, api_key=GOOGLE_API_KEY, num_images=NUM_IMAGES_PER_PROMPTS, score_threshold=IMG_SIMILARITY_THRESHOLD, aspect_ratio=ASPECT_RATIO)
    else:
        # img quality 1-4(time tradeoff)
        img_metadata_file = generate_images_flux(result, IMAGE_DIR, api_key=TOGETHER_AI_API_KEY, num_images=NUM_IMAGES_PER_PROMPTS, img_quality=4, aspect_ratio=ASPECT_RATIO)
    time_taken = calc_time(start)
    print(f"[INFO] Image Generation Completed in {time_taken}, saved to {IMAGE_DIR}")

    if stock_media == 'TRUE':
        print("[INFO] SCRAPING STOCK MEDIA")
        start = time.time()
        stock_media_json_path = scrape_stock_media(data_folder_path=VIDEO_SAVE_DIR,
                                                   json_file_path = file_path,
                                                   aspect_ratio = ASPECT_RATIO,
                                                   score_threshold = IMG_SIMILARITY_THRESHOLD,
                                                   api_key = GOOGLE_API_KEY,
                                                   pixaray_api_key = PIXARAY_API_KEY)
        time_taken = calc_time(start)
        print("[INFO] SCRAPING STOCK MEDIA COMPLETED IN", time_taken)
    else:
        print("[INFO] SCRAPING STOCK MEDIA - DISABLED")

    print("[INFO] TTS Started",TTS_MODEL)
    start = time.time()
    # Processing the script to remove the text and video cues
    processed_script = remove_enclosed_segments(result["script"])
    processed_script = processed_script.replace('<>', '')
    # Audio gen
    if TTS_MODEL == 'GEMINI':
        voice = 'Sulafat'
        try:
            save_file_path = generate_audio(script=processed_script, output_dir=AUDIO_DIR, api_key=GOOGLE_API_KEY,voice=voice)
        except:
            print('[ERROR] Failed to generate audio file - Trying AGAIN')
            save_file_path = generate_audio(script=processed_script, output_dir=AUDIO_DIR, api_key=GOOGLE_API_KEY,voice=voice)
    else:
        save_file_path = text_to_speech_file(text=processed_script, save_dir=AUDIO_DIR, ELEVENLABS_API_KEY=ELEVENLABS_API_KEY, AUDIO_SPEED=AUDIO_PACE) #eleven labs
    time_taken = calc_time(start)
    print(f"[INFO] TTS Completed in {time_taken}, saved to {save_file_path}")

    print("[INFO] CALCULATING SCENE and IMAGE TIMINGS",end = ' ')
    start = time.time()
    scene_timing_file = os.path.join(AUDIO_DIR, "scene_timings.json")
    fetch_audio_timings(audio_file=save_file_path,script_file=file_path,api_key=ASSEMBLYAI_API_KEY,output_file=scene_timing_file)
    img_timing_file = os.path.join(AUDIO_DIR, "img_timings.json")
    calculate_media_timings(scene_timing_file=scene_timing_file, img_metadata=img_metadata_file, output_file=img_timing_file)
    time_taken = calc_time(start)
    print(' - Completed in', time_taken)

    print("[INFO] VIDEO MERGING STARTED")
    start = time.time()
    video_path = os.path.join(VIDEO_SAVE_DIR, "merged_video.mp4")
    create_video_from_images(img_timing_file,IMAGE_DIR, save_file_path, video_path, TARGET_SIZE)
    time_taken = calc_time(start)
    print(f"[INFO] VIDEO MERGING COMPLETED IN {time_taken}, saved to:",video_path)

    print("[INFO] ADDING AVATAR")
    start = time.time()
    scale = 0.5
    avatar_video_path = os.path.join(VIDEO_SAVE_DIR, "avatar_video.mp4")
    add_avatar(audio_file=save_file_path, video_file=video_path, output_file=avatar_video_path, scale=scale)
    time_taken = calc_time(start)
    print(f"[INFO] AVATAR ADDED TO VIDEO AT A SCALE OF {scale}, saved to: ",avatar_video_path, "Time taken: ",time_taken)

    output_path = os.path.join(VIDEO_SAVE_DIR, "final_video.mp4")
    print("[INFO] GENERATING CAPTIONS")
    start = time.time()
    model_path = os.getenv("MODEL_PATH")
    generate_captions(avatar_video_path, output_path, model_path)
    time_taken = calc_time(start)
    print("[INFO] CAPTIONS GENERATED IN", time_taken)
    print(f"[INFO] FINAL VIDEO COMPLETED, saved to: {output_path}")

    if GENERATE_SHORTS == 'TRUE':
        print("[INFO] GENERATING SHORTS")
        start = time.time()
        SHORTS_DIR = os.path.join(VIDEO_SAVE_DIR, "shorts")
        os.makedirs(SHORTS_DIR, exist_ok=True)
        ASPECT_RATIO = "9:16"
        NUM_IMAGES_PER_PROMPTS = 2
        NUM_IMG_PROMPTS = 5
        AUDIO_PACE = 1.2
        MAX_VIDEO_LENGTH = 30
        TARGET_SIZE = [720, 1280]
        creator = ContentCreator(
            llm_provider=llm_provider,
            api_key=api_key,
            model_name=model_name,
            MAX_VIDEO_LENGTH=MAX_VIDEO_LENGTH, NUM_IMG_PROMPTS=NUM_IMG_PROMPTS, IMAGE_DIM=TARGET_SIZE, ASPECT_RATIO=ASPECT_RATIO
        )
        result = creator.create_content(topic, context=context, height=TARGET_SIZE[1], width=TARGET_SIZE[0],
                                        aspect_ratio=ASPECT_RATIO,time_unit='seconds')
        IMAGE_DIR = os.path.join(SHORTS_DIR, "images")
        os.makedirs(IMAGE_DIR, exist_ok=True)

        AUDIO_DIR = os.path.join(SHORTS_DIR, "audio")
        os.makedirs(AUDIO_DIR, exist_ok=True)
        filename = os.path.join(SHORTS_DIR, f"youtube_content_{topic.replace(' ', '_')[:20]}.json")
        file_path = creator.save_to_json(result, filename)
        if IMG_MODEL == 'GEMINI':
            img_metadata_file = generate_images(result, IMAGE_DIR, api_key=GOOGLE_API_KEY, num_images=NUM_IMAGES_PER_PROMPTS,
                            score_threshold=IMG_SIMILARITY_THRESHOLD, aspect_ratio=ASPECT_RATIO)
        else:
            img_metadata_file = generate_images_flux(result, IMAGE_DIR, api_key=TOGETHER_AI_API_KEY, num_images=NUM_IMAGES_PER_PROMPTS,
                                 img_quality=4, aspect_ratio=ASPECT_RATIO)
        processed_script = remove_enclosed_segments(result["script"])
        processed_script = processed_script.replace('<>', '')
        if TTS_MODEL == 'GEMINI':
            try:
                save_file_path = generate_audio(script=processed_script, output_dir=AUDIO_DIR, api_key=GOOGLE_API_KEY,
                                                voice='Umbriel')
            except:
                print('[ERROR] Failed to generate audio file - Trying AGAIN')
                save_file_path = generate_audio(script=processed_script, output_dir=AUDIO_DIR, api_key=GOOGLE_API_KEY,
                                                voice='Umbriel')
        else:
            save_file_path = text_to_speech_file(text=processed_script, save_dir=AUDIO_DIR,
                                                 ELEVENLABS_API_KEY=ELEVENLABS_API_KEY,
                                                 AUDIO_SPEED=AUDIO_PACE)  # eleven labs
        scene_timing_file = os.path.join(AUDIO_DIR, "scene_timings.json")
        fetch_audio_timings(audio_file=save_file_path, script_file=file_path, api_key=ASSEMBLYAI_API_KEY,
                            output_file=scene_timing_file)
        img_timing_file = os.path.join(AUDIO_DIR, "img_timings.json")
        calculate_media_timings(scene_timing_file=scene_timing_file,img_metadata=img_metadata_file,output_file=img_timing_file)
        video_path = os.path.join(SHORTS_DIR, "merged_video.mp4")
        create_video_from_images(IMAGE_DIR, save_file_path, video_path, TARGET_SIZE)
        avatar_video_path = os.path.join(SHORTS_DIR, "avatar_video.mp4")
        add_avatar(audio_file=save_file_path, video_file=video_path, output_file=avatar_video_path, scale=scale)
        output_path = os.path.join(VIDEO_SAVE_DIR, "shorts.mp4")
        generate_captions(avatar_video_path, output_path, model_path)

        time_taken = calc_time(start)
        print("[INFO] SHORTS GENERATED IN", time_taken)
    else:
        print("[INFO] GENERATING SHORTS - DISABLED")
    return result

if __name__ == "__main__":
    master_start = time.time()
    main()
    time_taken = calc_time(master_start)
    print("-------------------[INFO] THE SCRIPT COMPLETED IN", time_taken)