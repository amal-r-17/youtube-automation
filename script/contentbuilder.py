import os
import json
from typing import Dict, Any, Optional

from script.scriptGenerator import YouTubeContentGenerator

class ContentCreator:
    """
    Main class that handles YouTube content creation workflow.
    Generates scripts and image prompts and returns them in a structured format.
    """
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, model_name: Optional[str] = None, MAX_VIDEO_LENGTH=1, NUM_IMG_PROMPTS=10, IMAGE_DIM=[1280, 720], ASPECT_RATIO="16:9", SHORTS=False):
        """
        Initialize the ContentCreator with specified LLM provider.
        
        Args:
            llm_provider (str): The LLM provider to use ("openai", "grok", "gemini", "claude", "ollama")
            api_key (Optional[str]): API key for the chosen provider or base URL for Ollama
            model_name (Optional[str]): Specific model name to use with the provider
        """
        self.generator = YouTubeContentGenerator(
            llm_provider=llm_provider, 
            api_key=api_key,
            model_name=model_name,
            MAX_VIDEO_LENGTH=MAX_VIDEO_LENGTH, NUM_IMG_PROMPTS=NUM_IMG_PROMPTS, IMAGE_DIM=IMAGE_DIM, ASPECT_RATIO=ASPECT_RATIO,
            SHORTS=SHORTS
        )
        self.shorts = SHORTS
    def create_content(self, topic: str, context: str,height,width,aspect_ratio,time_unit='minutes') -> Dict[str, Any]:
        """
        Generate YouTube content based on the provided topic.
        
        Args:
            topic (str): The topic to create content about
            
        Returns:
            Dict[str, Any]: A dictionary containing the script and image prompts
        """
        # Step 1: Generate the YouTube script
        script = self.generator.generate_youtube_script(topic=topic, context=context,time_unit=time_unit)
        
        print("GENERATED SCRIPT: ", script)
        # Step 2: Generate image prompts from the script
        image_prompts = self.generator.generate_image_prompts(script=script,topic=topic,height=height,width=width,aspect_ratio=aspect_ratio)
        print("IMG prompts: ", image_prompts)
        # Step 3: Generate a Thumbnail from the script
        thumbnail_prompt = self.generator.generate_thumbnail_prompts(script)
        print("thumbnail_prompt: ", image_prompts)
        # Step 4: Generate a youtube video description
        video_description = self.generator.generate_youtube_description(script=script, topic=topic)

        # Step 3: Format the result
        result = {
            "topic": topic,
            "script": script,
            "image-prompts": image_prompts,
            "thumbnail-prompts": thumbnail_prompt,
            "meta_data": video_description
        }
        # print("Final result: ", result)
        # self.save_to_json(result=result, filename="sample.json")
        # exit()
        return result
    
    def save_to_json(self, result: Dict[str, Any], filename: str = "youtube_content.json") -> str:
        """
        Save the generated content to a JSON file.
        
        Args:
            result (Dict[str, Any]): The result from create_content
            filename (str): Name of the JSON file to save to
            
        Returns:
            str: Path to the saved file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        return os.path.abspath(filename)