from google import genai
from google.genai import types
import wave
import os
from pydub import AudioSegment
import io
from random import choice

def select_voice(voice_keyword = None):
    voice_keyword = voice_keyword.lower() if voice_keyword else None
    gemini_voices = {
        "male_voices": {
            "Zephyr": "Bright",
            "Puck": "Upbeat",
            "Charon": "Informative",
            "Orus": "Firm",
            "Fenrir": "Excitable",
            "Iapetus": "Clear",
            "Enceladus": "Breathy",
            "Algenib": "Gravelly",
            "Achernar": "Soft",
            "Alnilam": "Firm",
            "Schedar": "Even",
            "Gacrux": "Mature",
            "Rasalgethi": "Informative",
            "Achird": "Friendly",
            "Zubenelgenubi": "Casual",
            "Sadaltager": "Knowledgeable",
            "Umbriel": "Easy-going",
            "Algieba": "Smooth",
            "Sadachbia": "Lively",
            "Sulafat": "Warm"
        },
        "female_voices": {
            "Kore": "Firm",
            "Leda": "Youthful",
            "Aoede": "Breezy",
            "Callirrhoe": "Easy-going",
            "Autonoe": "Bright",
            "Despina": "Smooth",
            "Erinome": "Clear",
            "Laomedeia": "Upbeat",
            "Pulcherrima": "Forward",
            "Vindemiatrix": "Gentle"
        }
    }
    if voice_keyword is None:
        all_voices = sum([list(value.keys()) for value in gemini_voices.values()],[])
        voice = choice(all_voices)
    elif voice_keyword == 'male':
        all_voices = list(gemini_voices['male_voices'].keys())
        voice = choice(all_voices)
    elif voice_keyword == 'female':
        all_voices = list(gemini_voices['female_voices'].keys())
        voice = choice(all_voices)
    else:
        all_voices = sum([list(value.keys()) for value in gemini_voices.values()], [])
        all_voices = [voice.lower() for voice in all_voices]
        all_voice_types = sum([list(value.values()) for value in gemini_voices.values()], [])
        all_voice_types = [voice_type.lower() for voice_type in all_voice_types]
        if voice_keyword in all_voices:
            voice = voice_keyword
        elif voice_keyword in all_voice_types:
            all_voices = gemini_voices['male_voices']
            all_voices.update(gemini_voices['female_voices'])
            all_voices = [voice for voice in all_voices if all_voices[voice].lower() == voice_keyword]
            voice = choice(all_voices)
        else:
            print('Invalid voice keyword')
            all_voices = sum([list(value.keys()) for value in gemini_voices.values()], [])
            voice = choice(all_voices)
    return voice

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

def mp3_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    audio = AudioSegment(
        data=pcm,
        sample_width=sample_width,
        frame_rate=rate,
        channels=channels
    )
    audio.export(filename, format="mp3")

def generate_audio(script,api_key,output_dir,output_filename='audio_output.mp3',voice_keyword=None):
  client = genai.Client(api_key=api_key)
  voice = select_voice(voice_keyword)
  print('VOICE:', voice)
  response = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents=script,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=voice,
              )
          )
        ),
    )
  )
  output_file = os.path.join(output_dir, output_filename)
  data = response.candidates[0].content.parts[0].inline_data.data
  if output_filename.endswith('.mp3'):
      mp3_file(output_file, data)
  else:
      wave_file(output_file, data)
  print(f"{output_file}: A new audio file was saved successfully!")
  return output_file

# voices
# Zephyr -- Bright	Puck -- Upbeat	Charon -- Informative
# Kore -- Firm	Fenrir -- Excitable	Leda -- Youthful
# Orus -- Firm	Aoede -- Breezy	Callirrhoe -- Easy-going
# Autonoe -- Bright	Enceladus -- Breathy	Iapetus -- Clear
# Umbriel -- Easy-going	Algieba -- Smooth	Despina -- Smooth
# Erinome -- Clear	Algenib -- Gravelly	Rasalgethi -- Informative
# Laomedeia -- Upbeat	Achernar -- Soft	Alnilam -- Firm
# Schedar -- Even	Gacrux -- Mature	Pulcherrima -- Forward
# Achird -- Friendly	Zubenelgenubi -- Casual	Vindemiatrix -- Gentle
# Sadachbia -- Lively	Sadaltager -- Knowledgeable	Sulafat -- Warm

