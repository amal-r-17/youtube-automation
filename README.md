# YouTube Automation

Repository for a fully autonomous YouTube channel script generator.

## Script Generator

### Steps
1. Install requirements
2. Get the API key from your preferred service provider
3. Add the key and provider name (e.g., Gemini, OpenAI, Grok) to the `.env` file
4. Run the main file: `python main.py`

### Example
Run the following command to generate a script:
```bash
python main.py "What if NASA had the budget of the US military?"
```


# image-generator

This branch holds the module for generating images from prompts

Go to this link [https://aistudio.google.com/u/0/apikey](https://aistudio.google.com/u/0/apikey) and create a Gemini API key.

Create a config file with the code with the following structure

    image_gen = {
                 "GOOGLE_API_KEY":(YOUR API KEY)
                }
