from google import genai
from together import Together
from google.genai import types
from PIL import Image
from io import BytesIO
import requests
import os
import google.generativeai as gen_ai
import io
import numpy as np
from numpy.linalg import norm
from random import randint
import threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_image_data(image_path):
    """Loads an image from a file path and converts it to bytes."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    with Image.open(image_path) as img:
        # Convert to RGB if needed (e.g., if it's RGBA or grayscale) for consistency
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Convert PIL Image to bytes in a common format
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG') # Using JPEG for consistent MIME type
        return img_byte_arr.getvalue()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0 # Or raise an error, depending on desired behavior

    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def image_prompt_similarity_score(image_file_path,target_prompt,api_key,score_threshold=0.7):
    gen_ai.configure(api_key=api_key)
    vision_model = gen_ai.GenerativeModel('gemini-1.5-flash')
    embedding_model = gen_ai.GenerativeModel('models/embedding-001')
    try:
        image_data_bytes = load_image_data(image_file_path)
        caption_response = vision_model.generate_content([
          {
              'mime_type': 'image/jpeg',
              'data': image_data_bytes
          },
          "Generate a very concise, objective, descriptive caption for this image. Limit to 10-15 words, focusing on main subjects and their actions/colors."
        ])
        generated_caption = caption_response.text.strip()
        target_prompt_embedding_response = gen_ai.embed_content(
          model="models/embedding-001",
          content=target_prompt
        )
        generated_caption_embedding_response = gen_ai.embed_content(
          model="models/embedding-001",
          content=generated_caption
        )
        target_prompt_embedding = target_prompt_embedding_response['embedding'] # Access 'embedding' key
        generated_caption_embedding = generated_caption_embedding_response['embedding'] # Access 'embedding' key

        similarity_score = float(cosine_similarity(target_prompt_embedding, generated_caption_embedding))
        return similarity_score >= score_threshold, similarity_score
    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")
        print("Please ensure the image file path is correct.")
        return True,-1
    except gen_ai.types.APIError as api_err:
        print(f"Gemini API Error: {api_err}")
        print("Possible issues:")
        print("1. Your API key is incorrect or expired.")
        print("2. You have exceeded your API quota (check your Google Cloud Console / AI Studio dashboard).")
        print("3. There's a network issue or rate limit.")
        return True,-1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your input data and API setup.")
        return True,-1

def crop_image(img, aspect_ratio: str) -> bool:
    if not aspect_ratio:
        print("Error: 'aspect_ratio' must be provided for cropping.")
        return False

    try:
        original_width, original_height = img.size
        try:
            ratio_parts = aspect_ratio.split(':')
            if len(ratio_parts) != 2:
                raise ValueError("Aspect ratio must be in 'width:height' format (e.g., '16:9').")
            desired_ratio_w = int(ratio_parts[0])
            desired_ratio_h = int(ratio_parts[1])
            if desired_ratio_w <= 0 or desired_ratio_h <= 0:
                raise ValueError("Aspect ratio components must be positive integers.")

            desired_aspect = desired_ratio_w / desired_ratio_h
            current_aspect = original_width / original_height
            if round(current_aspect,1) == round(desired_aspect,1):
              print(f"Original image dimensions: {original_width}x{original_height}")
              print(f"Image is already in the {aspect_ratio} aspect ratio")
              return img
            if current_aspect > desired_aspect:
                # Original image is wider than the desired aspect ratio
                # Constrain by height, calculate new width
                calculated_target_height = original_height
                calculated_target_width = int(original_height * desired_aspect)
            else:
                # Original image is taller or has the same aspect ratio as desired
                # Constrain by width, calculate new height
                calculated_target_width = original_width
                calculated_target_height = int(original_width / desired_aspect)

            print(f"Cropping to aspect ratio {aspect_ratio}. Calculated target dimensions: {calculated_target_width}x{calculated_target_height}")

            print(f"Original image dimensions: {original_width}x{original_height}")
            print(f"Final crop dimensions: {calculated_target_width}x{calculated_target_height}")

            # Calculate the cropping box (left, upper, right, lower)
            # This centers the crop
            left = (original_width - calculated_target_width) / 2
            top = (original_height - calculated_target_height) / 2
            right = (original_width + calculated_target_width) / 2
            bottom = (original_height + calculated_target_height) / 2

            # Ensure coordinates are integers
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            # Ensure the crop box is within the image boundaries
            left = max(0, left)
            top = max(0, top)
            right = min(original_width, right)
            bottom = min(original_height, bottom)

            # Perform the crop
            cropped_img = img.crop((left, top, right, bottom))
            return cropped_img

        except ValueError as ve:
            print(f"Error parsing aspect ratio '{aspect_ratio}': {ve}")
            return False
    except Exception as e:
        print(f"An unexpected error occurred during image cropping: {e}")
        return False

def image_generator(prompt_str, api_key, model='gemini-2.0-flash-preview-image-generation', output_filepath='gemini-native-image.png',aspect_ratio='16:9'):
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
            # print(part.text)
            print('')
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image = crop_image(image, aspect_ratio=aspect_ratio)
            image.save(output_filepath)


# def generate_images(prompts, output_path, api_key, num_images=3, score_threshold=0.7, aspect_ratio='16:9'):
#   output = {"script": prompts['script'], "image-paths": {}}
#   image_count = 1
#   for i, prompt in prompts['image-prompts'].items():
#       print("IMG PROMPT #:",i)
#       # prompt = prompt + "  in aspect ratio 16:9"
#       for _ in range(num_images):
#         img_created = False
#         try:
#             output_file = f"{output_path}/{image_count}.png"
#             image_generator(prompt_str=prompt, output_filepath=output_file, api_key=api_key,aspect_ratio=aspect_ratio)
#             approve, score = image_prompt_similarity_score(image_file_path=output_file, target_prompt=prompt,api_key=api_key, score_threshold=score_threshold)
#             img_iter = 0
#             print(f"image #: {image_count}, img prompt #: {i}, score: {score}, image_iter: {img_iter}")
#             while not approve and img_iter <= 5:
#                 image_generator(prompt_str=prompt, output_filepath=output_file, api_key=api_key)
#                 approve, score = image_prompt_similarity_score(image_file_path=output_file, target_prompt=prompt,api_key=api_key, score_threshold=score_threshold)
#                 img_iter += 1
#                 print(f"image #: {image_count}, img prompt #: {i}, score: {score}, image_iter: {img_iter}")
#         except Exception as e:
#             print("ERROR OCCURED: ", e)
#             if img_created:
#                 print("IMAGE ALREADY CREATED for img prompt #:", image_count, "score:", score)
#             output_file = "No-image"
#         print("IMAGE SAVED to: ", output_file,"\n")
#         output["image-paths"][f"{image_count}"] = output_file
#         image_count+=1
#   return output

def generate_images(prompts, output_path, api_key, num_images=3, score_threshold=0.7, aspect_ratio='16:9', max_processes=4, rpm_limit=6):
    output = {"script": prompts['script'], "image-paths": {}}
    
    # Create a list of all image generation tasks
    tasks = []
    image_count = 1
    
    for i, prompt in prompts['image-prompts'].items():
        print("IMG PROMPT #:", i)
        for _ in range(num_images):
            tasks.append({
                'image_count': image_count,
                'prompt_id': i,
                'prompt': prompt,
                'output_path': output_path,
                'api_key': api_key,
                'score_threshold': score_threshold,
                'aspect_ratio': aspect_ratio
            })
            image_count += 1
    
    def generate_single_image(task):
        """Function to generate a single image with retry logic - runs in separate process"""
        image_count = task['image_count']
        prompt_id = task['prompt_id']
        prompt = task['prompt']
        output_path = task['output_path']
        api_key = task['api_key']
        score_threshold = task['score_threshold']
        aspect_ratio = task['aspect_ratio']
        
        img_created = False
        output_file = "No-image"
        score = 0
        
        try:
            output_file = f"{output_path}/{image_count}.png"
            image_generator(prompt_str=prompt, output_filepath=output_file, api_key=api_key, aspect_ratio=aspect_ratio)
            approve, score = image_prompt_similarity_score(image_file_path=output_file, target_prompt=prompt, api_key=api_key, score_threshold=score_threshold)
            img_iter = 0
            print(f"image #: {image_count}, img prompt #: {prompt_id}, score: {score}, image_iter: {img_iter}")
            
            while not approve and img_iter <= 5:
                image_generator(prompt_str=prompt, output_filepath=output_file, api_key=api_key, aspect_ratio=aspect_ratio)
                approve, score = image_prompt_similarity_score(image_file_path=output_file, target_prompt=prompt, api_key=api_key, score_threshold=score_threshold)
                img_iter += 1
                print(f"image #: {image_count}, img prompt #: {prompt_id}, score: {score}, image_iter: {img_iter}")
            
            img_created = True
            
        except Exception as e:
            print("ERROR OCCURED: ", e)
            if img_created:
                print("IMAGE ALREADY CREATED for img prompt #:", image_count, "score:", score)
            output_file = "No-image"
        
        print("IMAGE SAVED to: ", output_file, "\n")
        
        return image_count, output_file
    
    # Dictionary to store results
    results = {}
    
    # Process tasks in batches to respect RPM limit
    total_images = len(tasks)
    batch_size = rpm_limit
    
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"Processing batch: images {batch_start + 1}-{batch_end} ({len(batch_tasks)} images)")
        
        # Use min of batch size and max_processes for this batch
        actual_max_processes = min(max_processes, len(batch_tasks))
        
        with ThreadPoolExecutor(max_workers=actual_max_processes) as executor:
            # Submit batch tasks
            future_to_task = {executor.submit(generate_single_image, task): task for task in batch_tasks}
            
            # Wait for all tasks in this batch to complete
            for future in as_completed(future_to_task):
                try:
                    image_count, output_file = future.result()
                    results[image_count] = output_file
                except Exception as e:
                    task = future_to_task[future]
                    print(f"Task for image #{task['image_count']} failed with exception: {e}")
                    results[task['image_count']] = "No-image"
        
        # Wait before next batch (except for the last batch)
        if batch_end < total_images:
            print(f"Batch complete. Waiting 60 seconds for rate limit... ({total_images - batch_end} images remaining)")
            time.sleep(60)
    
    # Convert results to the expected format (string keys, ordered by image count)
    for i in range(1, len(tasks) + 1):
        if i in results:
            output["image-paths"][str(i)] = results[i]
        else:
            output["image-paths"][str(i)] = "No-image"
    
    return output

# def generate_images_flux(prompts, output_path, api_key, num_images=3, img_quality=4, aspect_ratio='16:9'):
#     output = {"script": prompts['script'], "image-paths": {}}
#     image_count = 1
#     client = Together(api_key=api_key)
#     width, height = 1280,720#1440, 816
#     if aspect_ratio == "9:16":
#         width, height = height,width
#     seed = randint(1,40)
#     for i, prompt in prompts['image-prompts'].items():
#         print("IMG PROMPT #:", i)
#         for _ in range(num_images):
#             seed = randint(1,40)
#             try:
#                 output_file = f"{output_path}/{image_count}.png"
#                 try:
#                     response = client.images.generate(
#                         prompt=prompt, model="black-forest-labs/FLUX.1-schnell-Free",
#                         width=width, height=height, steps=img_quality, seed=seed,
#                     )
#                     image_url = response.data[0].url
#                     img_data = requests.get(image_url).content
#                     with open(output_file, "wb") as f:
#                         f.write(img_data)
#                 except:
#                     seed = randint(1, 40)
#                     response = client.images.generate(
#                         prompt=prompt, model="black-forest-labs/FLUX.1-schnell-Free",
#                         width=width, height=height, steps=img_quality, seed=seed,
#                     )
#                     image_url = response.data[0].url
#                     img_data = requests.get(image_url).content
#                     with open(output_file, "wb") as f:
#                         f.write(img_data)
#             except Exception as e:
#                 print("ERROR OCCURED: ", e)
#                 output_file = "No-image"
#             print("IMAGE SAVED to: ", output_file, "\n")
#             output["image-paths"][f"{image_count}"] = output_file
#             image_count += 1
    # return output

def generate_images_flux(prompts, output_path, api_key, num_images=3, img_quality=4, aspect_ratio='16:9', max_threads=5, rpm_limit=6):
    output = {"script": prompts['script'], "image-paths": {}}
    
    # Create a list of all image generation tasks
    tasks = []
    image_count = 1
    
    for i, prompt in prompts['image-prompts'].items():
        print("IMG PROMPT #:", i)
        for _ in range(num_images):
            tasks.append({
                'image_count': image_count,
                'prompt_id': i,
                'prompt': prompt,
                'output_path': output_path,
                'api_key': api_key,
                'img_quality': img_quality,
                'aspect_ratio': aspect_ratio
            })
            image_count += 1
    
    # Thread-safe dictionary to store results
    results = {}
    results_lock = threading.Lock()
    
    def generate_single_flux_image(task):
        """Function to generate a single image using Flux API"""
        image_count = task['image_count']
        prompt_id = task['prompt_id']
        prompt = task['prompt']
        output_path = task['output_path']
        api_key = task['api_key']
        img_quality = task['img_quality']
        aspect_ratio = task['aspect_ratio']
        
        output_file = "No-image"
        
        # Create client instance for this thread
        client = Together(api_key=api_key)
        width, height = 1280, 720  # 1440, 816
        if aspect_ratio == "9:16":
            width, height = height, width
        
        try:
            output_file = f"{output_path}/{image_count}.png"
            seed = randint(1, 40)
            
            try:
                response = client.images.generate(
                    prompt=prompt, model="black-forest-labs/FLUX.1-schnell-Free",
                    width=width, height=height, steps=img_quality, seed=seed,
                )
                image_url = response.data[0].url
                img_data = requests.get(image_url).content
                with open(output_file, "wb") as f:
                    f.write(img_data)
            except:
                # Retry with different seed
                seed = randint(1, 40)
                response = client.images.generate(
                    prompt=prompt, model="black-forest-labs/FLUX.1-schnell-Free",
                    width=width, height=height, steps=img_quality, seed=seed,
                )
                image_url = response.data[0].url
                img_data = requests.get(image_url).content
                with open(output_file, "wb") as f:
                    f.write(img_data)
                    
        except Exception as e:
            print("ERROR OCCURED: ", e)
            output_file = "No-image"
        
        print("IMAGE SAVED to: ", output_file, "\n")
        
        # Thread-safe result storage
        with results_lock:
            results[image_count] = output_file
        
        return image_count, output_file
    
    # Process tasks in batches to respect RPM limit
    total_images = len(tasks)
    batch_size = rpm_limit - 1
    
    for batch_start in range(0, total_images, batch_size):
        st_time = time.time()
        batch_end = min(batch_start + batch_size, total_images)
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"Processing batch: images {batch_start + 1}-{batch_end} ({len(batch_tasks)} images)")
        
        # Use min of batch size and max_threads for this batch
        actual_max_threads = min(max_threads, len(batch_tasks))
        
        with ThreadPoolExecutor(max_workers=actual_max_threads) as executor:
            # Submit batch tasks
            future_to_task = {executor.submit(generate_single_flux_image, task): task for task in batch_tasks}
            
            # Wait for all tasks in this batch to complete
            for future in as_completed(future_to_task):
                try:
                    future.result()
                except Exception as e:
                    task = future_to_task[future]
                    print(f"Task for image #{task['image_count']} failed with exception: {e}")
        
        # Wait before next batch (except for the last batch)
        if batch_end < total_images:
            # time_passed = int(time.time() - st_time)
            # if time_passed<60:
            #     time_left = 60 - time_passed
            #     print(f"Batch complete. Waiting {time_left} seconds for rate limit... ({total_images - batch_end} images remaining)")
            #     time.sleep(time_left)
            # else:
            #     print("Starting the next batch")
            print(f"Batch complete. Waiting {60} seconds for rate limit... ({total_images - batch_end} images remaining)")
            time.sleep(60)
    
    # Convert results to the expected format (string keys, ordered by image count)
    for i in range(1, len(tasks) + 1):
        if i in results:
            output["image-paths"][str(i)] = results[i]
        else:
            output["image-paths"][str(i)] = "No-image"
    
    return output