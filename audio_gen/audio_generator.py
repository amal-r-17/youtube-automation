import json
import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv


# def generate_audio_from_json(json_file_path):
#     """
#     Reads text data from a JSON file and generates audio files using Eleven Labs API.

#     Args:
#         json_file_path (str): Path to the input JSON file.
#     """
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: JSON file not found at {json_file_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON format in {json_file_path}")
#         return

#     if not isinstance(data, list):
#         print("Error: JSON file should contain a list of objects.")
#         return

#     output_dir = "audio_outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     print(f"Processing {len(data)} entries from {json_file_path}...")

#     for i, entry in enumerate(data):
#         text = entry.get("text")
#         voice_id = entry.get("voice_id")
#         output_filename = entry.get("output_filename")
#         model_id = entry.get("model_id", "eleven_multilingual_v2") # Default model

#         if not text or not output_filename:
#             print(f"Skipping entry {i+1}: 'text' and 'output_filename' are required.")
#             continue

#         output_path = os.path.join(output_dir, output_filename)

#         print(f"Generating audio for '{text[:50]}...' (Voice ID: {voice_id if voice_id else 'default'}, Model: {model_id})")
#         try:
#             audio = generate(
#                 text=text,
#                 voice=voice_id,  # Pass voice_id directly
#                 model=model_id
#             )

#             with open(output_path, "wb") as f:
#                 f.write(audio)
#             print(f"Successfully saved audio to {output_path}")

#             # Optional: Play the audio immediately after generation
#             # play(audio)

#         except Exception as e:
#             print(f"Error generating audio for '{text[:50]}...': {e}")


load_dotenv("api.env")
key = os.getenv("ELEVENLABS_API_KEY")

if key is None:
    print("❌ API key not loaded!")
else:
    print("✅ API key loaded:", key[:5] + "...")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")



def text_to_speech_file(text: str, save_dir: str, ELEVENLABS_API_KEY, save_file_name="audio_output.mp3") -> str:
    elevenlabs = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)
    # Calling the text_to_speech conversion API with detailed parameters
    response = elevenlabs.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        # Optional voice settings that allow you to customize the output
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.1,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)

    # Generating a unique file name for the output MP3 file
    save_file_path = os.path.join(save_dir, save_file_name)#f'.\data\\audio\\audio_output.mp3'

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path



if __name__ == "__main__":
    json_file = ".\data\scripts\what_if_NASA_had_budget_of_US_miltary.json"
    file = json.load(open(json_file))
    text = file['script']
    text_to_speech_file(text)