import os, json
import re
import ast
from typing import Dict, List, Optional, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini models
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama


class YouTubeContentGenerator:
    """
    A class that generates YouTube scripts and corresponding image prompts using different LLM providers.
    
    This class can:
    1. Generate an engaging YouTube script based on a topic prompt
    2. Break down a script into sections and generate detailed image prompts for each section
    """
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, model_name: Optional[str] = None, MAX_VIDEO_LENGTH = 1,      NUM_IMG_PROMPTS=2, IMAGE_DIM=[1280,720], ASPECT_RATIO="16:9"):
        """
        Initialize the YouTubeContentGenerator with a specific LLM provider.
        
        Args:
            llm_provider (str): The LLM provider to use. Options: "openai", "grok", "gemini", "claude", "ollama"
            api_key (Optional[str]): The API key for the chosen provider. If None, will attempt to use
                                     environment variables.
            model_name (Optional[str]): Specific model name to use. If None, will use default for the provider.
        """
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.video_len = MAX_VIDEO_LENGTH
        self.num_img_prompts = NUM_IMG_PROMPTS
        self.segment_len = round((self.video_len*60)/self.num_img_prompts)
        self.IMAGE_DIM = IMAGE_DIM
        self.ASPECT_RATIO = ASPECT_RATIO

        
    def _initialize_llm(self):
        """Initialize the LLM based on the selected provider."""
        if self.llm_provider == "openai":
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            model = self.model_name or "gpt-4"
            return ChatOpenAI(temperature=0.7, model_name=model, api_key=api_key)
        
        elif self.llm_provider == "grok":
            api_key = self.api_key or os.environ.get("GROQ_API_KEY")
            model = self.model_name or "llama3-70b-8192"
            return ChatGroq(temperature=0.7, model_name=model, api_key=api_key)
        
        elif self.llm_provider == "gemini":
            api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            model = self.model_name or "gemini-2.0-flash"
            return ChatGoogleGenerativeAI(temperature=0.7, model=model, google_api_key=api_key)
        
        elif self.llm_provider == "claude":
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            model = self.model_name or "claude-3-opus-20240229"
            return ChatAnthropic(temperature=0.7, model_name=model, anthropic_api_key=api_key)
            
        elif self.llm_provider == "ollama":
            # For Ollama, we connect to a local instance
            # Default to localhost:11434 if no API key (which is used as base URL)
            base_url = self.api_key or "http://localhost:11434"
            model = self.model_name or "llama3"  # Default to llama3 if no model specified
            return ChatOllama(
                temperature=0.7,
                model=model,
                base_url=base_url
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. " 
                           f"Supported providers: openai, grok, gemini, claude, ollama")
    
    def generate_youtube_script(self, topic: str, context: str = "", time_unit: str = "minutes") -> str:
        """
        Generate an engaging YouTube script based on the provided topic.
        
        Args:
            topic (str): The topic or prompt for the YouTube video
            
        Returns:
            str: The generated YouTube script
        """
        script_prompt_template = """Create an engaging, curiosity-peaking YouTube script on the following topic:

                            TOPIC: {topic}

                            CONTEXT (if available): {context}
                            *Note: Use the provided context to inform your script with current, relevant information. If context is provided, prioritize these details and weave them naturally into your narrative. If no context is provided or context is empty, rely on your existing knowledge to create compelling content.*

                            Requirements:
                            - Divide this script into exactly {max_image_prompts} scenes, each approximately {segment_len} seconds of spoken content. Breaks should prioritize **natural narrative shifts** (e.g., change in subject, tone, or joke setup), but aim to stay close to the ~{segment_len}-second target for balance.
                            - The {max_image_prompts} scenes in the script must be separated by the symbol <>
                            - Don't add any text indicating scene index or youtube script
                            - The script should be for a video that's approximately {max_minute} {time_unit} long to allow for natural pacing, punchlines, and witty delivery.
                            - The **narrator's persona must be distinctively witty and sarcastic**. Think [MENTION A TYPE OF WIT/SARCASM OR A COMEDIC PERSONA, e.g., "dry and observational like a seasoned cynic," or "energetically sarcastic with a playful edge," or "darkly humorous and self-deprecating"]. Their commentary should be clever, perhaps a bit irreverent, but ultimately keep the audience hooked.
                            - Include a **strong, witty, and attention-grabbing hook** at the beginning. This should immediately establish the sarcastic tone and make viewers desperate to know what's next.
                            - The script must have a clear structure:
                                - **Witty Introduction:** Sets the stage, introduces the topic with the narrator's unique, sarcastic perspective, and perhaps pokes fun at common misconceptions or the topic itself.
                                - **Main Content (3-4 Key Points):** Each point should be presented with the narrator's signature wit and sarcasm. This is where you'll build arguments, share facts, or tell stories, all filtered through their cynical or humorous lens. *If context is provided, integrate relevant information from the context naturally into these points.*
                                - **Witty Conclusion:** Summarizes the main (perhaps absurd) takeaways, possibly with a final sarcastic jab or an unexpected twist, leading smoothly into the call-to-action.
                            - **Strategically integrate 2-3 "Spike Moments" within the main content.** These are segments designed to be particularly memorable, funny, or insightful due to a sharp comedic observation, an unexpected sarcastic turn, a surprisingly blunt truth delivered with wit, or a moment of heightened (perhaps mock) drama. These should be identifiable *through the narrator's dialogue and delivery style* as implied by the script's wording, creating natural peaks in engagement. For example, the narrator might say something like, "And here's the kicker, the part where your jaw officially meets the floor..." or deliver a particularly cutting line.
                            - **When context is available:** 
                                - Seamlessly blend web-sourced information with your narrative voice
                                - Use current examples, recent developments, or fresh perspectives from the context
                                - Reference specific details, statistics, or quotes that enhance the story
                                - Maintain the witty tone even when presenting factual information from the context
                            - **When context is not available:**
                                - Draw from your existing knowledge base to create compelling content
                                - Focus on timeless aspects of the topic that don't require recent updates
                                - Use well-established facts, historical examples, or universal truths about the subject
                            - Use language that **actively creates curiosity and leverages the witty/sarcastic tone** to keep viewers entertained and watching. Wordplay, irony, and humorous understatements are encouraged.
                            - **ABSOLUTELY NO technical directions** like [PAUSE], [CUT TO], [SOUND EFFECT], [VISUAL DESCRIPTION], or any bracketed instructions for editing or voice acting. The script must *only* contain the words to be spoken.
                            - Ensure the script **flows naturally when read aloud**, maintaining the witty, sarcastic, and conversational style throughout. It should sound like someone genuinely (and hilariously) expressing these thoughts.
                            - The script should not include any cues for background music, sound effects, or visual elements on screen.
                            - Include an **engaging and characterful call-to-action** at the end, delivered in the narrator's witty/sarcastic style. It should feel like a natural extension of their personality (e.g., "So, if you enjoy a good dose of reality, however unpalatable, hit that subscribe button. Or don't. More existential dread for me, I guess.").
                            - Make sure that the actual script doesn't include any mentioning of background sound, or specifying the spike moments or spike points, it should only be the final script that's it.
                            - **CRITICAL: The output must ONLY contain the spoken dialogue.** Do not include:
                                - Section headers like "Introduction," "Main Content," "Conclusion"
                                - Point numbers like "Point 1," "Point 2," etc.
                                - Labels like "Hook:" or "Call-to-Action:"
                                - Any meta-commentary about the script structure
                                - Transition markers or formatting indicators
                            - **Start immediately with the opening hook dialogue** and flow seamlessly through to the final call-to-action without any structural labels or breaks.
                            - The response should read like a continuous monologue that could be spoken directly into a microphone.

                            YOUTUBE SCRIPT:
                """
        
        script_prompt = PromptTemplate(
            input_variables=["topic", "max_minute", "context","max_image_prompts","segment_len","time_unit"],
            template=script_prompt_template
        )
        
        script_chain = LLMChain(llm=self.llm, prompt=script_prompt)
        result = script_chain.run(topic=topic, max_minute=self.video_len, context=context,max_image_prompts=self.num_img_prompts,segment_len=self.segment_len,time_unit=time_unit)
        return result.strip()
    
    def generate_image_prompts(self, script: str, topic: str,height,width,aspect_ratio) -> Dict[str, str]:
        section_prompt_template = """
        You are given a YouTube narration script for a scene. Your task is to:
    
        SCENE: {scene}

        Generate an **extremely detailed, literal, and visually grounded** image prompt suitable for image generation AI, fully aligned with the narration.

        IMPORTANT: Follow these critical guidelines:

        A. **Absolutely No Metaphors or Visual Abstractions**
        - Do NOT create prompt for figurative phrases, visual metaphors, or surreal representations.
        - Only portray literal, concrete subjects, actions, or locations described or strongly implied in the narration.
        - The image should be related to the context of {topic}
        - For example, if the script says, “I was drowning in work,” the image should show a cluttered workspace, NOT a person submerged in a sea of documents.
        - All image prompts must be based in **observable, physical scenes or actions**.

        B. **No Facial Close-Ups to Depict Emotion or Curiosity**
        - If people appear, show them in medium or wide shots.
        - Emotions should be conveyed subtly through posture, body language, or scene composition—not via close-up expressions.

        C. **Faithful Visual Representation of the Script**
        - Each image must accurately reflect what is happening in that portion of the script. Avoid inserting additional interpretations, jokes, or embellishments not directly supported by the text.
        - Prioritize key actions, settings, physical objects, or characters that are explicitly or implicitly present in that part of the narration.

        D. **Image Prompt Structure :**

        Return your response as a text which is a full detailed image prompt for that scene. The image prompt must include:

        - **Main Subject/Focus:** What’s the key visual or action being described? Ground it in the script.
        - **Key Characters/Objects (if any):** Describe visible figures or important objects. Avoid focusing on facial expressions.
        - **Background and Setting:** Indoors/outdoors, location, time of day, and relevant context. Detailed or suggestive background based on script tone.
        - **Lighting, Mood, and Atmosphere:** E.g., bright daylight, office gloom, chaotic kitchen, sarcastic tension, etc.
        - **Implied Perspective/Composition:** Suggested camera angle or framing—medium shot, overhead, wide, etc.—avoiding metaphorical/emotive close-ups.
        - **Style Suggestion:** Photorealistic, cartoonish, cinematic, blueprint, vector, gritty comic, etc. Suit style to tone of narration.
        - **Color Palette Suggestions:** Vibrant, desaturated, pastel, moody shadows, high contrast, muted tones, etc.
        - **Must-Have Visual Elements:** 2–3 tangible items, places, or actions **explicitly described or strongly implied** in the script segment.
        - **Image Dimensions & Aspect Ratio:** Ensure strict adherence to: WIDTH: {WIDTH}, HEIGHT: {HEIGHT}, Aspect ratio: {ASPECT_RATIO}
        - **Image must have 4k picture quality

        ⚠️ **DO NOT** include any bracketed or editorial comments like [SHOW SCENE OF…] or [LIGHTNING IN BACKGROUND]. Only return the image prompts as final spoken scene descriptions.

        **OUTPUT FORMAT:**

        Return the image prompt in text format
        """

        section_prompt = PromptTemplate(
            input_variables=["scene","topic", "WIDTH", "HEIGHT", "ASPECT_RATIO"],
            template=section_prompt_template
        )

        scenes = [scene.strip('\n') for scene in script.split('<>')]
        section_chain = LLMChain(llm=self.llm, prompt=section_prompt)
        final_prompts = {}
        for index, scene in enumerate(scenes):
            resp = section_chain.run(scene=scene,
                                     topic=topic,
                                     WIDTH=width,
                                     HEIGHT=height,
                                     ASPECT_RATIO=aspect_ratio)
            final_prompts[f'{index + 1}'] = resp
        # final_prompts = self.parse_image_prompts(final_prompts)
        return final_prompts
        
    

    def parse_image_prompts(self, result_text):
        """
        Parse the LLM output and return a proper dictionary with keys 1-6 and image prompts as values.
        
        Args:
            result_text (str): The raw output from the LLM
            
        Returns:
            Dict[int, str]: A dictionary mapping section numbers to detailed image prompts
        """
        # Clean up the result
        result = result_text.strip()
        
        # Remove code block markers if present
        if result.startswith("```python"):
            result = result.split("```python")[1]
        if result.startswith("```"):
            result = result.split("```")[1]
        if result.endswith("```"):
            result = result.rsplit("```", 1)[0]
        
        result = result.strip()
        
        # Try using ast.literal_eval (safe evaluation of Python literals)
        try:
            parsed = ast.literal_eval(result)
            # Ensure keys are integers
            return {int(k): str(v) for k, v in parsed.items()}
        except:
            pass
        
        # Manual parsing as fallback
        try:
            prompts = {}
            
            # Find dictionary content between braces
            start = result.find('{')
            end = result.rfind('}')
            if start != -1 and end != -1:
                dict_content = result[start+1:end]
                
                # Split by lines and parse each entry
                lines = dict_content.split('\n')
                current_key = None
                current_value = []
                
                for line in lines:
                    line = line.strip()
                    if not line or line == ',':
                        continue
                    
                    # Check if line starts with a number (new key)
                    key_match = re.match(r'^(\d+):\s*"(.*)$', line)
                    if key_match:
                        # Save previous entry
                        if current_key is not None and current_value:
                            prompts[current_key] = ' '.join(current_value).rstrip('",')
                        
                        # Start new entry
                        current_key = int(key_match.group(1))
                        current_value = [key_match.group(2)]
                    else:
                        # Continue current value
                        if current_key is not None:
                            current_value.append(line.rstrip('",'))
                
                # Save last entry
                if current_key is not None and current_value:
                    prompts[current_key] = ' '.join(current_value).rstrip('",')
            
            if len(prompts) >= 6:
                return prompts
                
        except Exception as e:
            print(f"Manual parsing failed: {e}")
        
        # Final fallback - return error dictionary
        return {
            i+1: f"Failed to parse section {i+1}. Please review the script and try again."
            for i in range(6)
        }
        

    def generate_thumbnail_prompts(self, script: str) -> Dict[str, Dict[str, str]]:
        """
        Generate 3 different thumbnail options for a YouTube script.
        
        Args:
            script (str): The YouTube script to process
            
        Returns:
            Dict[str, Dict[str, str]]: A dictionary with 3 options, each containing 'text' and 'image_prompt'
        """
        
        section_prompt_template = """
        
        I'm going to give you a YouTube script. Analyze the script to understand the main topic, key points, and overall theme, then create 3 DIFFERENT highly detailed image prompts for generating clickbait thumbnails that will maximize clicks.

        YOUTUBE SCRIPT:
        {script}

        Create 3 DIFFERENT extremly detailed thumbnail concepts, each with a unique approach:

        **OPTION 1 - DRAMATIC/SHOCK VALUE:**
        Focus on dramatic facial expressions, bold contrasts, and shock value elements.

        **OPTION 2 - CURIOSITY/MYSTERY:**
        Focus on creating curiosity gaps, mystery elements, and "what will happen" scenarios.

        **OPTION 3 - BENEFIT/RESULT:**
        Focus on the end result, transformation, or key benefit from the video content.

        For each option, include:

        **VISUAL ELEMENTS:**
        - Main subject/focal point that represents the core topic
        - Eye-catching visual metaphors or symbols related to the content
        - Dramatic facial expressions (if people are involved) showing appropriate emotion
        - Bold, contrasting elements that create visual tension

        **CLICKBAIT FACTORS:**
        - Vivid, saturated colors (especially reds, yellows, blues for high contrast)
        - Dynamic composition with elements that draw the eye
        - Visual elements that match the option's theme
        - Dramatic lighting and shadows for maximum impact

        **TECHNICAL SPECIFICATIONS:**
        - 16:9 aspect ratio optimized for YouTube thumbnails
        - High contrast and bold elements that remain visible at small sizes
        - Clear focal hierarchy with the most important element dominating
        - Background that doesn't compete with the main subject

        **TEXT OVERLAY INTEGRATION:**
        - Include bold, eye-catching text overlay directly burned into the image
        - Text should be 2-4 words maximum, extracted from the most compelling part of the script
        - Use large, thick fonts (like Impact or Bebas Neue style) in bright colors (white, yellow, or red)
        - Position text strategically to not obscure the main visual elements
        - Add drop shadows, outlines, or background boxes to ensure text readability

        Format your response EXACTLY as:

        **OPTION 1:**
        **TEXT:** [2-4 word phrase]
        **IMAGE PROMPT:** [Complete detailed image generation prompt]

        **OPTION 2:**
        **TEXT:** [2-4 word phrase]
        **IMAGE PROMPT:** [Complete detailed image generation prompt]

        **OPTION 3:**
        **TEXT:** [2-4 word phrase]
        **IMAGE PROMPT:** [Complete detailed image generation prompt]

        """
        
        section_prompt = PromptTemplate(
            input_variables=["script"],
            template=section_prompt_template
        )
        
        section_chain = LLMChain(llm=self.llm, prompt=section_prompt)
        result = section_chain.run(script=script)

        # print("output: ", result)
        parsed_options = self.parse_multiple_thumbnail_outputs(result)
        # print("final: ", parsed_options)
        
        return parsed_options


    def parse_multiple_thumbnail_outputs(self, output_text):
        """
        Parse multiple thumbnail outputs and return dict with 3 options.
        """
        result = {
            "option_1": {"text": "", "image_prompt": ""},
            "option_2": {"text": "", "image_prompt": ""},
            "option_3": {"text": "", "image_prompt": ""}
        }
        
        # Split by OPTION sections
        sections = output_text.split("**OPTION")
        
        for section in sections[1:]:  # Skip first empty section
            option_num = None
            
            # Get option number from the beginning of section
            if section.strip().startswith("1") or " 1:" in section[:10]:
                option_num = "option_1"
            elif section.strip().startswith("2") or " 2:" in section[:10]:
                option_num = "option_2"
            elif section.strip().startswith("3") or " 3:" in section[:10]:
                option_num = "option_3"
            
            if option_num:
                # Find TEXT
                if "**TEXT:**" in section:
                    text_start = section.find("**TEXT:**") + len("**TEXT:**")
                    text_end = section.find("\n", text_start)
                    if text_end == -1:
                        text_end = section.find("**IMAGE PROMPT:**", text_start)
                    if text_end == -1:
                        text_end = len(section)
                    
                    text_content = section[text_start:text_end].strip()
                    result[option_num]["text"] = text_content.replace('*', '').strip()
                
                # Find IMAGE PROMPT
                if "**IMAGE PROMPT:**" in section:
                    prompt_start = section.find("**IMAGE PROMPT:**") + len("**IMAGE PROMPT:**")
                    # Get everything after IMAGE PROMPT until end of section
                    prompt_content = section[prompt_start:].strip()
                    result[option_num]["image_prompt"] = prompt_content.replace('*', '').strip()
        
        return result
    
    def generate_youtube_description(self, script: str, topic: str) -> Dict[str, str]:
        """
        Generate an SEO-optimized YouTube video description based on script and topic.
        
        Args:
            script (str): The YouTube script content
            topic (str): The main topic/theme of the video
            
        Returns:
            Dict[str, str]: A dictionary containing 'description', 'tags', and 'title_suggestions'
        """
        
        description_prompt_template = """
        
        I'm going to give you a YouTube script and topic. Create a simple, engaging YouTube video description similar to popular science channels.

        YOUTUBE SCRIPT:
        {script}

        VIDEO TOPIC:
        {topic}

        Create a concise YouTube description with this structure:

        **MAIN_DESCRIPTION:**
        Write 1 paragraph (80-120 words) that:
        - Starts with an engaging question related to the topic
        - Briefly explains what the video covers
        - Mentions 2-3 key points from the script
        - Includes natural keywords
        - Ends with a simple call-to-action like "Don't forget to like, share, and subscribe for more [topic type] videos!"

        **HASHTAGS:**
        Generate 8-12 relevant hashtags including the main topic and related keywords

        **TITLE_SUGGESTIONS:**
        Provide 3 clickbait-style title options (under 60 characters each)

        Format your response EXACTLY as:

        **MAIN_DESCRIPTION:**
        [Single engaging paragraph with question opener, key points, and CTA]

        **HASHTAGS:**
        [8-12 hashtags separated by spaces]

        **TITLE_SUGGESTIONS:**
        1. [Title option 1]
        2. [Title option 2] 
        3. [Title option 3]

        """
        
        description_prompt = PromptTemplate(
            input_variables=["script", "topic"],
            template=description_prompt_template
        )
        
        description_chain = LLMChain(llm=self.llm, prompt=description_prompt)
        result = description_chain.run(script=script, topic=topic)

        # print("Description output: ", result)
        parsed_description = self.parse_youtube_description_output(result)

        return parsed_description


    def parse_youtube_description_output(self, output_text):
        """
        Parse simple YouTube description output and return structured dict.
        """
        result = {
            "main_description": "",
            "hashtags": "",
            "title_suggestions": [],
            "full_description": ""
        }
        
        sections = {
            "**MAIN_DESCRIPTION:**": "main_description",
            "**HASHTAGS:**": "hashtags",
            "**TITLE_SUGGESTIONS:**": "title_suggestions"
        }
        
        # Parse each section
        for section_header, key in sections.items():
            if section_header in output_text:
                start_idx = output_text.find(section_header) + len(section_header)
                
                # Find the end of this section (next section header or end of text)
                end_idx = len(output_text)
                for other_header in sections.keys():
                    if other_header != section_header and other_header in output_text:
                        other_start = output_text.find(other_header)
                        if other_start > start_idx and other_start < end_idx:
                            end_idx = other_start
                
                section_content = output_text[start_idx:end_idx].strip()
                
                # Special handling for title suggestions
                if key == "title_suggestions":
                    titles = []
                    for line in section_content.split('\n'):
                        line = line.strip()
                        if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                            title = line.split('.', 1)[1].strip() if '.' in line else line
                            titles.append(title)
                    result[key] = titles
                else:
                    result[key] = section_content.replace('*', '').strip()
        
        # Create full description (main description + hashtags)
        full_desc_parts = []
        
        if result["main_description"]:
            full_desc_parts.append(result["main_description"])
        
        if result["hashtags"]:
            full_desc_parts.append(" " + result["hashtags"])
        
        result["full_description"] = "".join(full_desc_parts)
        
        return result