from crewai import LLM
from crewai.tools import tool


llm = LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434")


class SearchTools():

  @tool("Search the internet")
  def search_internet(query):
    """Useful to search the internet
    about a a given topic and return relevant results"""
    prompt = (
        "You are a knowledgeable research assistant. Answer the following "
        "query with the most accurate, detailed, and up-to-date information "
        "you have. Provide specific facts, figures, names, and details. "
        "Structure your response as a list of key findings.\n\n"
        f"Query: {query}"
    )
    return llm.call(prompt)
