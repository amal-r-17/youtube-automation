import asyncio
# Corrected import statement for google.generativeai
from google import genai
from google.genai import types
import json
import logging
import contextlib
import wave
import os
import math
from pydub import AudioSegment
from pydub.generators import Sine  # Although Sine is imported, it's not used in the provided code.
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

# Configure logging for better debugging of the API calls
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BGM_Generator')
logger.setLevel(logging.DEBUG)


# Renamed to avoid confusion with the context manager version
def write_wave_file_pcm(filename, pcm, channels=1, rate=24000, sample_width=2):
    """
    Writes raw PCM audio data to a WAV file.
    This function is not directly used in the current flow after context manager was introduced.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def get_bgm_list(transcript_file, audio_file, api_key):
    logger.info(f"Getting BGM list for transcript: {transcript_file}")
    transcript = json.load(open(transcript_file))
    text = " ".join([item["text"] for item in transcript])  # Concatenate all text for prompt

    audio = AudioSegment.from_mp3(audio_file)
    duration_seconds = len(audio) / 1000.0
    logger.info(f"Audio total duration: {duration_seconds:.2f} seconds")

    client = genai.Client(api_key=api_key)

    # Upload the audio file to the Google GenAI file service
    # logger.info(f"Uploading audio file: {audio_file}")
    try:
        myfile = client.files.upload(file=audio_file)
        # logger.info(f"Audio file uploaded successfully. File ID: {myfile.name}")
    except Exception as e:
        logger.error(f"Error uploading audio file: {e}")
        raise

    prompt = """
    You are given an audio clip from a YouTube video. Your task is to segment the audio into distinct parts based on changes in tone, mood, or scene transitions.

    For each segment, suggest a suitable background music (BGM) that enhances the emotional and narrative flow.
    
    Requirements for the BGM across segments:
    - All BGMs must share a **cohesive musical style and sonic palette** (e.g., same instrument family, tempo range, key, and pitch)
    - Avoid drastic genre or mood shifts between segments — ensure a **smooth and seamless transition** when concatenated
    - Think of it like a film score: each part is unique but fits into a **unified, continuous background soundscape**
    
    For each segment, return an object with the following:
    - `"label"`: A short (3–5 word) descriptive name for the background music (e.g., "warm ambient texture", "subtle rhythmic pulse")
    - `"description"`: A 1–2 sentence explanation describing the mood, instrumentation, and emotional purpose of the BGM — all within the shared overall style
    - `"start_time"`: Segment start time in seconds
    - `"end_time"`: Segment end time in seconds
    
    Constraints:
    - The sum of all segment durations must **exactly equal {duration_seconds}**
    - Ensure **natural pacing and progression** across segments
    - All background tracks must be musically compatible — suitable for **direct concatenation without abrupt changes**
    
    Output Format: A valid JSON array of objects.
    
    Example:
    [
      {{
        "label": "warm ambient intro",
        "description": "A soft ambient pad with light textures, setting a reflective tone while staying minimal.",
        "start_time": 0,
        "end_time": 12
      }},
      {{
        "label": "evolving ambient swell",
        "description": "A gentle continuation with added atmospheric textures and subtle rhythmic movement, maintaining the ambient feel.",
        "start_time": 12,
        "end_time": 38
      }}
    ]
    """

    logger.info("Sending prompt to Gemini model for BGM suggestions...")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt.format(duration_seconds=duration_seconds), myfile]
    )
    raw_text = response.text.strip()

    # Step 2: Remove Markdown code block if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]  # remove the first line ```json
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    bgm_list = json.loads(raw_text)
    # print(bgm_list)
    for i in range(len(bgm_list) - 1):
        current_end = bgm_list[i]['end_time']
        next_start = bgm_list[i + 1]['start_time']
        diff = next_start - current_end
        if diff > 0.1:
            bgm_list[i]['end_time'] = bgm_list[i]['end_time'] + diff
        bgm_list[-1]['end_time'] = duration_seconds
        bgm_list[i]['duration'] = bgm_list[i]['end_time'] - bgm_list[i]['start_time']
    bgm_list[-1]['duration'] = bgm_list[-1]['end_time'] - bgm_list[-1]['start_time']

    transcript = json.load(open(transcript_file))
    scene_end_times = [scene['end_time'] for scene in transcript]
    scene_end_times_1 = []
    for end_time in scene_end_times:
        minutes, seconds = end_time.split(":")
        minutes, seconds = int(minutes), int(seconds)
        time_seconds = minutes * 60 + seconds
        scene_end_times_1.append(time_seconds)

    for i in range(len(bgm_list)):
        bgm = bgm_list[i]
        diff = [abs(end_time - bgm['end_time']) for end_time in scene_end_times_1]
        end_time = scene_end_times_1[diff.index(min(diff))]
        bgm['end_time'] = end_time
        if i != len(bgm_list) - 1:
            bgm_list[i + 1]['start_time'] = end_time
            bgm['duration'] = bgm['end_time'] - bgm['start_time']
    return bgm_list


@contextlib.contextmanager
def wave_file_context(filename, channels=2, rate=48000, sample_width=2):
    """
    A context manager for safely opening and writing to a WAV file.
    Ensures the file is properly closed even if errors occur.
    """
    # logger.debug(f"Opening WAV file context for: {filename}")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
    # logger.debug(f"Closing WAV file context for: {filename}")

def parse_input(input_text):
    """
    Parses a string of weighted prompts into a list of WeightedPrompt objects.
    Supports single prompt or comma-separated "text:weight" format.
    This function is primarily for interactive input, not directly used by generate_bgm loop.
    """
    logger.debug(f"Parsing input prompt: '{input_text}'")
    if ":" in input_text:
        parsed_prompts = []
        segments = input_text.split(',')
        malformed_segment_exists = False  # Tracks if any segment had a parsing error

        for segment_str_raw in segments:
            segment_str = segment_str_raw.strip()
            if not segment_str:  # Skip empty segments (e.g., from "text1:1, , text2:2")
                continue

            # Split on the first colon only, in case prompt text itself contains colons
            parts = segment_str.split(':', 1)

            if len(parts) == 2:
                text_p = parts[0].strip()
                weight_s = parts[1].strip()

                if not text_p:
                    logger.warning(f"Empty prompt text in segment '{segment_str_raw}'. Skipping this segment.")
                    malformed_segment_exists = True
                    continue
                try:
                    weight_f = float(weight_s)
                    parsed_prompts.append(types.WeightedPrompt(text=text_p, weight=weight_f))
                except ValueError:
                    logger.warning(
                        f"Invalid weight '{weight_s}' in segment '{segment_str_raw}'. Must be a number. Skipping this segment.")
                    malformed_segment_exists = True
                    continue
            else:
                logger.warning(f"Segment '{segment_str_raw}' is not in 'text:weight' format. Skipping this segment.")
                malformed_segment_exists = True
                continue

        if parsed_prompts:
            prompt_repr = [f"'{p.text}':{p.weight}" for p in parsed_prompts]
            # if malformed_segment_exists:
            #     logger.info(
            #         f"Partially sending {len(parsed_prompts)} valid weighted prompt(s) due to errors in other segments: {', '.join(prompt_repr)}")
            # else:
            #     logger.info(f"Sending multiple weighted prompts: {', '.join(prompt_repr)}")
            return parsed_prompts
        else:
            # logger.error(
            #     "Input contained ':' suggesting multi-prompt format, but no valid 'text:weight' segments were successfully parsed. No action taken.")
            return None
    else:
        # logger.info(f"Sending single text prompt: \"{input_text}\"")
        return [types.WeightedPrompt(text=input_text, weight=1.0)]  # Return as a list for consistency


async def generate_music(api_key, bgm_folder, prompts, max_chunks=19):

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    MODEL_ID = 'models/lyria-realtime-exp'
    config = types.LiveMusicGenerationConfig()

    # Use the global file_index from generate_music function attribute
    global_file_index = generate_music.file_index
    output_filename = os.path.join(bgm_folder, f"audio_{global_file_index}.wav")
    generate_music.file_index += 1  # Increment for the next call

    # logger.info(f"Generating music for prompt: '{prompts[0].text}' to {output_filename}")
    try:
        async with client.aio.live.music.connect(model=MODEL_ID) as session:
            async def receive_audio_chunks():
                with wave_file_context(output_filename) as wav:
                    chunk_count = 0
                    async for message in session.receive():
                        chunk_count += 1
                        if chunk_count > max_chunks:
                            # logger.info(f"Reached max_chunks ({max_chunks}). Stopping audio reception.")
                            break
                        audio_chunk = message.server_content.audio_chunks[0].data
                        if audio_chunk:
                            # logger.debug(f"Received audio chunk {chunk_count}/{max_chunks}")
                            wav.writeframes(audio_chunk)
                        else:
                            logger.warning("Received empty audio chunk.")

            await session.set_weighted_prompts(prompts=prompts)

            if config is not None:
                await session.set_music_generation_config(config=config)

            # logger.info("Starting music generation...")
            await session.play()

            # Start the task to receive audio chunks
            receive_task = asyncio.create_task(receive_audio_chunks())
            await receive_task  # Wait for the reception task to complete

            # logger.info("Music generation and reception completed for this segment.")
    except Exception as e:
        logger.error(f"Error during music generation for '{prompts[0].text}': {e}")
        # Optionally re-raise or handle more gracefully
        raise

    return output_filename


def trim_audio_segment(audio_segment: AudioSegment, target_duration_seconds: float) -> AudioSegment:
    """
    Trims or pads an audio segment to a target duration.
    Padding is done with silence if the segment is shorter than the target.
    Trimming occurs if the segment is longer.
    """
    target_duration_seconds = math.ceil(target_duration_seconds)  # Round up to ensure enough length

    if audio_segment.duration_seconds < target_duration_seconds:
        padding_needed_ms = (target_duration_seconds - audio_segment.duration_seconds) * 1000
        # Ensure frame rate matches the audio segment for padding
        silent_padding = AudioSegment.silent(duration=padding_needed_ms,
                                             frame_rate=audio_segment.frame_rate)
        trimmed_segment = audio_segment + silent_padding
        # logger.info(
        #     f"  Padded audio from {audio_segment.duration_seconds:.2f}s to {trimmed_segment.duration_seconds:.2f}s (target {target_duration_seconds:.2f}s).")
    else:
        trimmed_segment = audio_segment[:target_duration_seconds * 1000]
        # logger.info(
        #     f"  Trimmed audio from {audio_segment.duration_seconds:.2f}s to {trimmed_segment.duration_seconds:.2f}s (target {target_duration_seconds:.2f}s).")
    return trimmed_segment


async def process_and_merge_bgms(bgm_data: list[dict], output_filename: str = "merged_bgm.wav",
                                 bgm_folder: str = "BGM", target_volume_percentage = 20):
    logger.info("--- Starting BGM Processing and Merging Process ---")
    trimmed_bgm_segments = []

    for file_index, item in enumerate(bgm_data):
        text = item['description']
        duration = item['duration']
        if file_index > 0:
            duration += 1 # 1 second for overlapping with previous track ending
        # Construct path to the generated individual BGM file
        bgm_file_path = os.path.join(bgm_folder, f"audio_{file_index}.wav")

        # logger.info(f"\nProcessing BGM {file_index + 1}/{len(bgm_data)}: '{text}' (Target duration: {duration:.2f}s)")
        # logger.info(f"  Attempting to load: {bgm_file_path}")
        try:
            if not os.path.exists(bgm_file_path):
                # logger.error(f"  Error: BGM file not found at {bgm_file_path}. Skipping this item.")
                continue  # Skip to the next item if file not found

            generated_audio = AudioSegment.from_file(bgm_file_path, format="wav")
            trimmed_audio = trim_audio_segment(generated_audio, duration)
            fading_TP = (duration*1000)/100*30 #30% time period
            fading_min = 2500
            fade_in_duration_ms = math.ceil(min(fading_TP,fading_min))
            if file_index > 0:
                fade_in_duration_ms += 1000 #for fade in overlap
            fading_TP = (duration*1000)/100*20 #20% time period  #if fadeout is less than fade in it's better
            fade_out_duration_ms = math.ceil(min(fading_TP, fading_min-1000))
            faded_audio = trimmed_audio.fade_in(fade_in_duration_ms).fade_out(fade_out_duration_ms)
            trimmed_bgm_segments.append(faded_audio)
            # logger.info(
            #     f"Finished processing BGM {file_index + 1}. Trimmed length: {trimmed_audio.duration_seconds:.2f}s")

        except Exception as e:
            logger.error(f"Error processing BGM '{text}' from '{bgm_file_path}': {e}")

    if not trimmed_bgm_segments:
        logger.warning("\nNo BGM segments were successfully processed to merge. Nothing to do.")
        return

    logger.info("\n--- Merging all trimmed BGM segments ---")
    merged_bgm = AudioSegment.empty()
    if trimmed_bgm_segments:
        # Initialize with the frame rate of the first segment
        merged_bgm = AudioSegment.silent(duration=0, frame_rate=trimmed_bgm_segments[0].frame_rate)

    for i,segment in enumerate(trimmed_bgm_segments):
        if i > 0:
            fade_in_overlap = fading_min - 1000
            segment_fade_in_overlap = segment[:fade_in_overlap]
            segment = segment[fade_in_overlap:]
            overlay_position = max(0, len(merged_bgm) - fade_in_overlap)
            merged_bgm = merged_bgm.overlay(segment_fade_in_overlap,position=overlay_position)
        merged_bgm += segment

    logger.info(f"All BGMs merged successfully! Total duration: {merged_bgm.duration_seconds:.2f}s")

    amplitude_ratio = target_volume_percentage / 100.0
    if amplitude_ratio == 0:
        volume_reduction_db = math.inf  # Essentially mute
        print("Setting volume to 0% (mute).")
    else:
        volume_reduction_db = -(20 * math.log10(amplitude_ratio))

    merged_bgm = merged_bgm - volume_reduction_db
    logger.info(f"Reduced final BGM volume to {target_volume_percentage}%, reduced by {volume_reduction_db} dB, ")

    try:
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
        merged_bgm.export(output_filename, format="wav")
        # logger.info(f"Merged BGM saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        logger.error(f"Error saving merged BGM to file '{output_filename}': {e}")
        raise
    # logger.info("--- Process Completed ---")


async def generate_bgm(api_key, transcript_file, audio_file,target_volume_percentage):
    logger.info(f"Starting overall BGM generation for {audio_file}")
    try:
        bgm_list = get_bgm_list(transcript_file, audio_file, api_key)
    except:
        print("[ERROR] Failed to get BGM list from transcript, Trying AGAIN")
        bgm_list = get_bgm_list(transcript_file, audio_file, api_key)
    # Create a dedicated BGM subfolder for generated segments
    bgm_folder = os.path.join(os.path.dirname(audio_file), 'BGM')
    os.makedirs(bgm_folder, exist_ok=True)
    logger.info(f"Individual BGM segments will be saved in: {bgm_folder}")

    # Initialize a counter for filenames within generate_music to ensure unique names
    generate_music.file_index = 0

    for bgm_item in tqdm(bgm_list, desc="Generating individual BGM segments"):
        # Call generate_music for each BGM item
        # The prompts should be a list of WeightedPrompt objects
        await generate_music(
            api_key=api_key,
            bgm_folder=bgm_folder,
            max_chunks=19,  # Max chunks to receive per BGM segment
            prompts=[types.WeightedPrompt(text=bgm_item['description'], weight=1.0)]
        )

    # After generating all individual BGM files, process and merge them
    final_output_path = os.path.join(bgm_folder, "BGM.wav")
    await process_and_merge_bgms(bgm_list, output_filename=final_output_path, bgm_folder=bgm_folder,
                                 target_volume_percentage = target_volume_percentage)

    # logger.info(f"Overall BGM generation completed. Final output: {final_output_path}")
    return final_output_path

async def combine_audio_files(audio_file,bgm_file):
    bgm = AudioSegment.from_wav(bgm_file)
    audio = AudioSegment.from_mp3(audio_file)
    combined_audio = audio.overlay(bgm, position=0)
    output_file = os.path.join(os.path.dirname(audio_file), 'final_audio.mp3')
    combined_audio = combined_audio[:audio.duration_seconds * 1000]
    combined_audio.export(output_file, format='mp3')
    return output_file

# Main execution block
async def add_bgm(API_KEY,transcript_file,audio_file,target_volume_percentage):
    # API_KEY = "AIzaSyA5Rc_4NoMy5ALZ3eoDzgfQ5pCESJuSOlM"
    # dummy_audio_dir = "data/Dying_on_Every_Planet!/audio"
    # os.makedirs(dummy_audio_dir, exist_ok=True)
    # dummy_transcript_path = os.path.join(dummy_audio_dir, "scene_timings.json")
    # dummy_audio_path = os.path.join(dummy_audio_dir, "audio_output.mp3")
    # transcript_file = dummy_transcript_path
    # audio_file = dummy_audio_path

    bgm_file = await generate_bgm(api_key=API_KEY, transcript_file=transcript_file, audio_file=audio_file,target_volume_percentage=target_volume_percentage)
    logger.info(f"Final BGM file location: {bgm_file}")
    audio_file = await combine_audio_files(audio_file=audio_file, bgm_file=bgm_file)
    logger.info(f"Final Audio file location: {audio_file}")
    return audio_file