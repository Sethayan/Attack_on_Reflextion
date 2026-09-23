from crewai import Agent, LLM
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools


# --- Model constants (Change 3) ---
# Travel Concierge (hub/manager) and Self-Reflection use the reasoning model.
# All four spoke agents use the smaller model to control cost/latency.
REASONING_LLM = LLM(model="ollama/qwen3:14b", base_url="http://localhost:11434")
SPOKE_LLM = LLM(model="ollama/qwen2.5:7b", base_url="http://localhost:11434")


class TripAgents():

    # --- Tool references (one per agent, per Change 1.3) ---
    search_tool = SearchTools.search_internet
    browser_tool = BrowserTools.scrape_and_summarize_website
    calculator_tool = CalculatorTools.calculate

    def city_selection_agent(self):
        return Agent(
            role='City Selection Expert',
            goal='Select the best city based on weather, season, and prices',
            backstory=
            'An expert in analyzing travel data to pick ideal destinations',
            tools=[self.search_tool],
            llm=SPOKE_LLM,
            allow_delegation=False,
            verbose=True)

    def local_expert(self):
        return Agent(
            role='Local Expert at this city',
            goal='Provide the BEST insights about the selected city',
            backstory="""A knowledgeable local guide with extensive information
            about the city, it's attractions and customs""",
            tools=[self.search_tool, self.calculator_tool],
            llm=SPOKE_LLM,
            allow_delegation=False,
            verbose=True)

    def weather_agent(self):
        return Agent(
            role="Weather Agent",
            goal="Provide accurate weather forecasts for the destination and travel dates",
            backstory="An expert at interpreting forecast data and translating it into "
                       "practical packing and scheduling advice.",
            tools=[self.browser_tool],
            llm=SPOKE_LLM,
            allow_delegation=False,
            verbose=True,
        )

    def messaging_agent(self):
        return Agent(
            role="Messaging Agent",
            goal="Compose clear, accurate user-facing updates and confirmations",
            backstory="Responsible for all outgoing communication to the traveler — "
                       "booking confirmations, itinerary summaries, and alerts.",
            tools=[],
            llm=SPOKE_LLM,
            allow_delegation=False,
            verbose=True,
        )

    def travel_concierge(self):
        return Agent(
            role='Amazing Travel Concierge',
            goal="""Create the most amazing travel itineraries with budget and 
            packing suggestions for the city""",
            backstory="""Specialist in travel planning and logistics with 
            decades of experience""",
            tools=[],   # Manager agent cannot have tools in hierarchical mode
            llm=REASONING_LLM,
            # No allow_delegation=False — this is the hierarchical manager
            verbose=True)