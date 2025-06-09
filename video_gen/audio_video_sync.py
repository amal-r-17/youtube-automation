import assemblyai as aai
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import re
import json
from langchain.chains import LLMChain
from num2words import num2words

def convert_numbers_in_string(text):
    # Find all numbers and replace them with their word equivalent
    return re.sub(r'\d+', lambda match: num2words(int(match.group())), text)

def remove_enclosed_segments(text):
    return re.sub(r"\*\*\([^)]*\)\*\*", "", text)

def transcribe(audio_file,api_key):
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    transcript = aai.Transcriber(config=config).transcribe(audio_file)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    return transcript

def create_timed_subtitles(transcript):
    subtitles = []
    i = 0
    sentence_dict = {}
    for text in transcript.export_subtitles_srt().split('\n')[:-1]:
        if i % 4 == 3:
            subtitles.append(sentence_dict)
            sentence_dict = {}
        if i % 4 == 0:
            sentence_dict["index"] = int(text)
        if i % 4 == 1:
            start, end = text.split(' --> ')
            sentence_dict["start_time"] = pd.Timestamp(start).strftime('%M:%S')
            sentence_dict["end_time"] = pd.Timestamp(end).strftime('%M:%S')
        if i % 4 == 2:
            sentence_dict["text"] = text
        i += 1
    return subtitles

def clean_script(script):
    script_processed = remove_enclosed_segments(script)
    script_processed = script_processed.replace('\n', '')
    script_processed = script_processed.lower().replace('-', ' ').replace('*', ' ')
    script_processed = re.sub(r"[()\—\.,\?“”’:]", "", script_processed)
    script_processed = convert_numbers_in_string(script_processed)
    return script_processed

def create_scene_timings(script_processed,subtitles):
    scene_timings = []
    last_subtitle_index = 0
    scenes = script_processed.split('<>')
    for i, scene in enumerate(scenes):
        scene_timing = {}
        words_scene = scene.split(' ')
        part_of_scene = False
        for subtitle in subtitles:
            if subtitle["index"] > last_subtitle_index:
                subtitle_text = subtitle["text"].lower().replace('-', ' ')
                words_subtitle = subtitle_text.split(' ')
                words_subtitle = [re.sub(r"[()\—\.,\?“”’:*]", "", word) for word in words_subtitle]
                if len([word for word in words_subtitle if word in words_scene]) / len(words_subtitle) > 0.6:
                    # part of scene
                    if part_of_scene == False:
                        part_of_scene = True
                        scene_timing = {
                            "index": i + 1,
                            "text": scene,
                            "start_time": subtitle['start_time']
                        }
                if part_of_scene == True and len([word for word in words_subtitle if word in words_scene]) / len(words_subtitle) <= 0.6:
                    part_of_scene = False
                    scene_timing['end_time'] = subtitle['start_time']
                    last_subtitle_index = subtitle['index'] - 1
                    break
        scene_timings.append(scene_timing)
    return scene_timings

def adjust_scene_timings(scene_timings,subtitles):
    # adjusting timings
    scene_timings[-1]['end_time'] = subtitles[-1]['end_time']
    last_scene_end_timing = scene_timings[0]['end_time']
    for scene in scene_timings[1:]:
        if scene['start_time'] != last_scene_end_timing:
            scene['start_time'] = last_scene_end_timing
        last_scene_end_timing = scene['end_time']

    return scene_timings

def fetch_audio_timings(audio_file,script_file,api_key,output_file):
    transcript = transcribe(audio_file,api_key)
    subtitles = create_timed_subtitles(transcript)
    script_json = json.load(open(script_file))
    script = script_json['script']
    script_processed = clean_script(script)
    scene_timings = create_scene_timings(script_processed,subtitles)
    scene_timings = adjust_scene_timings(scene_timings,subtitles)
    json.dump(scene_timings, open(output_file, "w"))