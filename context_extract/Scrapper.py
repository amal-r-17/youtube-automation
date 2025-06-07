import os
import newspaper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class ContentSummarizer:
    def __init__(self, min_len: int = 100, api_key=None):
        # self.topic = topic
        self.min_len = min_len
        self.retrieved_urls = []
        self.retrieved_titles = []
        self.scraped_contents = []
        self.number_of_sources = 0

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

    def search_duckduckgo(self, topic):
        search = DuckDuckGoSearchResults(output_format="list")
        search_results = search.invoke(topic)

        for result in search_results:
            self.retrieved_urls.append(result['link'])
            self.retrieved_titles.append(result['title'])

        self.number_of_sources = len(self.retrieved_urls)

    def extract_articles(self):
        results = []
        for url in self.retrieved_urls:
            try:
                article = newspaper.Article(url, language='en')
                article.download()
                article.parse()
                text = article.text
                if len(text) >= self.min_len:
                    results.append(text)
            except Exception as e:
                print(f"[Newspaper3k] Failed to process: {url} | {e}")
        self.scraped_contents = results

    def summarize(self, topic) -> str:
        prompt_template = PromptTemplate(
            input_variables=["topic", "number_of_sources", "titles", "contents"],
            template="""
                    You are an expert in analyzing content and distilling it into engaging, concise points.

                    We are creating a YouTube video on the topic: **{topic}**. To support this, we have gathered content from {number_of_sources} sources. Below are the titles:  
                    {titles}  
                    And their corresponding contents:  
                    {contents}

                    Your task is to extract and summarize the key insights into **at least 5 well-structured and compelling points**. These points should:
                    - Be highly informative and relevant to the topic.
                    - Capture the essence of all the provided content.
                    - Be suitable for use in a YouTube script to engage viewers.

                    Focus on clarity, impact, and completeness of information.
                    """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        return chain.run(
            topic=topic,
            number_of_sources=self.number_of_sources,
            titles=self.retrieved_titles,
            contents=self.scraped_contents
        )
    
    def run(self, topic=""):
        self.search_duckduckgo(topic=topic)
        self.extract_articles()
        content = self.summarize(topic=topic)
        return content


# ─────────────────────────────────────────────────────────────
# 🏁 Driver Function
# ─────────────────────────────────────────────────────────────
# def driver():
#     topic = "Vijay Mallya and RCB"
#     summarizer = ContentSummarizer(topic)
#     summarizer.search_duckduckgo()
#     summarizer.extract_articles()
#     summary = summarizer.summarize()

#     with open(f"main_points_{topic}.txt", "w", encoding="utf-8") as f:
#         f.write(summary)

#     print("✅ Summary saved to file.")
#     return summary

# # Run when this script is executed
# if __name__ == "__main__":
#     output = driver()
#     print("\n🔎 Final Output:\n", output)