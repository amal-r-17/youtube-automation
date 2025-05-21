import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()  # take environment variables
from script.contentbuilder import ContentCreator

def main():
    """
    Main function to demonstrate the content creation workflow.
    """
    # Get configuration from environment variables or use defaults
    llm_provider = os.getenv("LLM_TYPE", "gemini")
    
    # Handle API key or base URL based on provider
    api_key = None
    if llm_provider == "ollama":
        api_key = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        api_key_name = f"{llm_provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_name)
    
    # Get model name if specified
    model_name = os.environ.get(f"{llm_provider.upper()}_MODEL")
    
    # Instantiate the content creator
    creator = ContentCreator(
        llm_provider=llm_provider,
        api_key=api_key,
        model_name=model_name
    )
    
    # Get topic from command line arguments or use a default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "what if NASA had the budjet of US military?"
    
    print(f"Generating YouTube content for topic: '{topic}'")
    print(f"Using LLM provider: {llm_provider}")
    
    # Generate content
    result = creator.create_content(topic)
    
    # Print the result
    print("\n=== Generated Script ===")
    print(result["script"])
    
    print("\n=== Generated Image Prompts ===")
    for section_num, prompt in result["image-prompts"].items():
        print(f"Section {section_num}:")
        print(prompt)
        print()
    
    # Save to file
    filename = f"youtube_content_{topic.replace(' ', '_')[:20]}.json"
    file_path = creator.save_to_json(result, filename)
    print(f"\nContent saved to: {file_path}")
    
    return result


if __name__ == "__main__":
    main()