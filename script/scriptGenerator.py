import os
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
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, model_name: Optional[str] = None):
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
    
    def generate_youtube_script(self, topic: str) -> str:
        """
        Generate an engaging YouTube script based on the provided topic.
        
        Args:
            topic (str): The topic or prompt for the YouTube video
            
        Returns:
            str: The generated YouTube script
        """
        script_prompt_template = """
        Create an engaging, curiosity-peaking YouTube script on the following topic:
        
        TOPIC: {topic}
        
        Requirements:
        - The script should be for a video that's at least 1 minute long
        - Make it engaging, conversational, and with a strong hook at the beginning
        - Include an introduction, main content with 3-4 key points, and a conclusion
        - Use language that creates curiosity and keeps viewers watching
        - Don't include any technical directions like [PAUSE] or [CUT TO]
        - Make sure the script flows naturally when read aloud
        - Make sure that the script doesn't include any cues for the actual video itself, like how the background sound should be or what image to be shown then or anything, it should only have the actual script
        - Include an engaging call-to-action at the end
        
        YOUTUBE SCRIPT:
        """
        
        script_prompt = PromptTemplate(
            input_variables=["topic"],
            template=script_prompt_template
        )
        
        script_chain = LLMChain(llm=self.llm, prompt=script_prompt)
        result = script_chain.run(topic=topic)
        # print("result: ", result)
        # exit()
        return result.strip()
    
    def generate_image_prompts(self, script: str) -> Dict[int, str]:
        """
        Break down a YouTube script into 6 sections and generate detailed image prompts for each.
        
        Args:
            script (str): The YouTube script to process
            
        Returns:
            Dict[int, str]: A dictionary mapping section numbers to detailed image prompts
        """
        # First, we need to break down the script into 6 roughly equal parts
        section_prompt_template = """
        I'm going to give you a YouTube script. Break this script into exactly 6 equal parts (approximately 10 seconds each when spoken).
        
        YOUTUBE SCRIPT:
        {script}
        
        For each part, create a highly detailed image prompt that would perfectly illustrate what's being said in that section.
        The image prompts should be extremely detailed, including:
        - The main subject/focus
        - Background elements and setting
        - Lighting, mood, and atmosphere
        - Style suggestion (photorealistic, illustrated, etc.)
        - Color palette suggestions
        - Any specific visual elements that support the narration
        
        Format your response exactly as a Python dictionary with keys 1-6 and detailed image prompts as values:
        
        {{
            1: "First detailed image prompt...",
            2: "Second detailed image prompt...",
            3: "Third detailed image prompt...",
            4: "Fourth detailed image prompt...",
            5: "Fifth detailed image prompt...",
            6: "Sixth detailed image prompt..."
        }}
        
        Make sure your response can be directly parsed as a Python dictionary. Include ONLY the dictionary in your response.
        """
        
        section_prompt = PromptTemplate(
            input_variables=["script"],
            template=section_prompt_template
        )
        
        section_chain = LLMChain(llm=self.llm, prompt=section_prompt)
        result = section_chain.run(script=script)
        
        # Clean up the result to make sure it's parsable
        result = result.strip()
        if result.startswith("```python"):
            result = result.split("```python")[1]
        if result.startswith("```"):
            result = result.split("```")[1]
        if result.endswith("```"):
            result = result.rsplit("```", 1)[0]
        
        # WARNING: Using eval() is generally not recommended for security reasons
        # In a production environment, you should use a proper JSON parser with error handling
        try:
            image_prompts = eval(result)
            return image_prompts
        except:
            # Fallback method if parsing fails
            return {
                i+1: f"Failed to parse section {i+1}. Please review the script and try again."
                for i in range(6)
            }