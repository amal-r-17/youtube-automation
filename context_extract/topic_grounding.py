from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

def grounding_gemini(api_key,topic):
  client = genai.Client(api_key=api_key)
  model_id = "gemini-2.0-flash"

  google_search_tool = Tool(
      google_search = GoogleSearch()
  )

  response = client.models.generate_content(
      model=model_id,
      contents=topic,
      config=GenerateContentConfig(
          tools=[google_search_tool],
          response_modalities=["TEXT"],
      )
  )

  output = '\n\n'.join(each.text for each in response.candidates[0].content.parts)
  return output

