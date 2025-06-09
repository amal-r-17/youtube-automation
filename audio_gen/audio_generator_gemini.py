from google import genai
from google.genai import types
import wave
import os
from pydub import AudioSegment
import io

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

def generate_audio(script,api_key,output_dir,output_filename='audio_output.mp3',voice='Gacrux'):
  client = genai.Client(api_key=api_key)

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