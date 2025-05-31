import os, re
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()  # take environment variables
from script.contentbuilder import ContentCreator
from image_gen.image_gen import generate_images  # Gen images
from audio_gen.audio_generator import text_to_speech_file
from video_gen.video_generator import create_video_from_images
from video_gen.caption_generator import generate_captions


def remove_enclosed_segments(text):
    return re.sub(r"\*\*\([^)]*\)\*\*", "", text)


def main():
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
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    SAVE_DIR = os.getenv("SAVE_DIR")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Video size
    # TARGET_SIZE = (1280, 720) #(1920, 1080)

    # VIDEO PARAMS
    MAX_LENTH_MINS = int(os.getenv("MAX_LENTH_MINS", 1))
    NUM_IMAGE_PROMPTS = int(os.getenv("NUM_IMAGE_PROMPTS", 10))
    NUM_IMAGES_PER_PROMPTS = int(os.getenv("NUM_IMAGES_PER_PROMPTS", 2))
    TARGET_SIZE = [int(val) for val in os.getenv("VIDEO_DIMENSION", "1280,720").split(",")]
    ASPECT_RATIO = os.getenv("ASPECT_RATIO", "16:9")

    # IMAGE GEN
    IMG_SIMILARITY_THRESHOLD = float(os.getenv("IMG_SIMILARITY_THRESHOLD", 0.7))

    # AUDIO
    GENERATE_CAPTIONS = bool(os.getenv("GENERATE_CAPTIONS", True))
    AUDIO_PACE = float(os.getenv("AUDIO_PACE", 1.1))

    # Instantiate the content creator
    creator = ContentCreator(
        llm_provider=llm_provider,
        api_key=api_key,
        model_name=model_name,
        MAX_VIDEO_LENGTH=MAX_LENTH_MINS, NUM_IMG_PROMPTS=NUM_IMAGE_PROMPTS, IMAGE_DIM=TARGET_SIZE,
        ASPECT_RATIO=ASPECT_RATIO
    )

    # Get topic from command line arguments or use a default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "what if NASA had the budget of US military?"

    print(f"Generating YouTube content for topic: '{topic}'")
    print(f"Using LLM provider: {llm_provider}")

    # Generate content
    result = creator.create_content(topic)

    # Getting the title for saving dir
    result = creator.create_content(topic)
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
    print("\n=== Generated Script ===")
    print(result["script"])

    print("\n=== Generated Image Prompts ===")
    for section_num, prompt in result["image-prompts"].items():
        print(f"Section {section_num}:")
        print(prompt)

    # Save to file
    filename = os.path.join(VIDEO_SAVE_DIR, f"youtube_content_{topic.replace(' ', '_')[:20]}.json")
    file_path = creator.save_to_json(result, filename)
    print(f"\nContent saved to: {file_path}")

    # exit()

    print("[INFO] Image Generation Started")
    # Gen images
    generate_images(result, IMAGE_DIR, api_key=GOOGLE_API_KEY)
    print("[INFO] Image Generation Completed")

    print("[INFO] TTS Started")

    # Processing the script to remove the text and video cues
    processed_script = remove_enclosed_segments(result["script"])
    # Audio gen
    save_file_path = text_to_speech_file(text=processed_script, save_dir=AUDIO_DIR, ELEVENLABS_API_KEY=ELEVENLABS_API_KEY)
    print("[INFO] TTS Completed")

    print("[INFO] VIDEO MERGING STARTED")
    video_path = os.path.join(VIDEO_SAVE_DIR, "merged_video.mp4")
    create_video_from_images(IMAGE_DIR, save_file_path, video_path, TARGET_SIZE)
    print("[INFO] VIDEO MERGING COMPLETED, saved to:",video_path)
    output_path = os.path.join(VIDEO_SAVE_DIR, "final_video.mp4")
    print("[INFO] GENERATING CAPTIONS")
    model_path = os.getenv("MODEL_PATH")
    generate_captions(video_path, output_path, model_path)
    print("[INFO] CAPTIONS GENERATED")
    print(f"[INFO] FINAL VIDEO COMPLETED, saved to: {output_path}")

    return result
if __name__ == "__main__":
    main()