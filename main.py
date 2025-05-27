import os, re
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()  # take environment variables
from script.contentbuilder import ContentCreator
from image_gen.image_gen import generate_images # Gen images
from audio_gen.audio_generator import text_to_speech_file
from video_gen.video_generator import create_video_from_images

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

    # Image directory
    IMAGE_DIR = os.getenv("IMAGE_DIR")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Audio directory
    AUDIO_DIR = os.getenv("AUDIO_DIR")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Video size
    TARGET_SIZE = (1280, 720) #(1920, 1080)


    # Instantiate the content creator
    creator = ContentCreator(
        llm_provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )
    
    # Get topic from command line arguments or use a default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "what if NASA had the budget of US military?"
    
    print(f"Generating YouTube content for topic: '{topic}'")
    print(f"Using LLM provider: {llm_provider}")
    
    # Generate content
    result = creator.create_content(topic)
    
    # Print the result
    print("\n=== Generated Script ===")
    print(result["script"])
    
    print("\n=== Generated Image Prompts ===")
    for section_num, prompt in result["image-prompts"].items():
        print(f"Section {section_num}:")
        print(prompt)
        print()
    
    # Save to file
    filename = f"youtube_content_{topic.replace(' ', '_')[:20]}.json"
    file_path = creator.save_to_json(result, filename)
    print(f"\nContent saved to: {file_path}")

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

    print("[INFO] FINAL VIDEO STARTED")
    # Generate video
    create_video_from_images(IMAGE_DIR, save_file_path, "sample.mp4", TARGET_SIZE)
    print("[INFO] FINAL VIDEO COMPLETED")

    return result


if __name__ == "__main__":
    main()