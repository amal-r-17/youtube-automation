from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
# from config import image_gen
import json
import os

def image_generator(prompt_str, api_key, model='gemini-2.0-flash-preview-image-generation', output_filepath='gemini-native-image.png'):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt_str,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(output_filepath)


def generate_images(prompts, output_path, api_key, num_images=3):
  output = {"script": prompts['script'], "image-paths": {}}
  image_count = 1
  for i, prompt in prompts['image-prompts'].items():
      print("PROCESSING IMAGE: ", prompt)
      for _ in range(num_images):
        try:
            output_file = f"{output_path}/{image_count}.png"
            image_generator(prompt_str=prompt, output_filepath=output_file, api_key=api_key)
        except Exception as e:
            print("ERROR OCCURED: ", e)
            output_file = "No-image"
        print("IMAGE SAVED to: ", output_file)
        output["image-paths"][f"{image_count}"] = output_file
        image_count+=1
  return output


if __name__ == "__main__":
  input_file = 'data/input/what_if_NASA_had_budget_of_US_miltary.json'
  output_folder = input_file.split('/')[-1].split('.json')[0]
  output_path = f"data/output/{output_folder}/"
  if not os.path.exists(output_path):
      os.mkdir(output_path)
  prompts = json.load(open(input_file))

  output = generate_images(prompts, output_path)

  json.dump(output, open(f"{output_path}{output_folder}_image_paths.json", 'w'))