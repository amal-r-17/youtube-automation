from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from config import image_gen

def image_generator(prompt,model='gemini-2.0-flash-preview-image-generation',output_file = 'gemini-native-image.png'):
    client = genai.Client(api_key=image_gen['GOOGLE_API_KEY'])
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
          response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
      if part.text is not None:
        print(part.text)
      elif part.inline_data is not None:
        image = Image.open(BytesIO((part.inline_data.data)))
        image.save(output_file)

image_generator(prompt='A vibrant and surreal scene depicting the Thrissur Pooram festival in Kerala, '
                       'India, but instead of elephants, majestic dinosaurs of various sizes and species are adorned with '
                       'traditional Nettipattam (ornamental head coverings) and carrying people on howdahs. The dinosaurs are amidst a large, '
                       'enthusiastic crowd of people in traditional Kerala attire. The iconic Vadakkunnathan Temple stands tall and '
                       'illuminated in the background. Bright fireworks explode in the twilight sky, adding to the festive atmosphere. '
                       'Capture the energy, color, and unique juxtaposition of ancient tradition and prehistoric creatures.')