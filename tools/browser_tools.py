import requests
from crewai import LLM
from crewai.tools import tool
from unstructured.partition.html import partition_html


llm = LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434")


class BrowserTools():

  @tool("Scrape website content")
  def scrape_and_summarize_website(website):
    """Useful to scrape and summarize a website content"""
  
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    }
    response = requests.get(website, headers=headers, timeout=15)
    elements = partition_html(text=response.text)
    content = "\n\n".join([str(el) for el in elements])
    content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
    summaries = []
    for chunk in content:
      prompt = (
          "Analyze and summarize the content below, make sure to include "
          "the most relevant information in the summary, return only the "
          f"summary nothing else.\n\nCONTENT\n----------\n{chunk}"
      )
      summary = llm.call(prompt)
      summaries.append(summary)
    return "\n\n".join(summaries)
