from crewai import Task
from textwrap import dedent
from datetime import date
import builtins
import re

class TripTasks:

    def identify_task(self, agent, origin, cities, interests, range, extra_context=""):
        return Task(
            description=dedent(f"""\
                {extra_context}
                Analyze and select the best city for the trip based 
                on specific criteria such as weather patterns, seasonal
                events, and travel costs. This task involves comparing
                multiple cities, considering factors like current weather
                conditions, upcoming cultural or seasonal events, and
                overall travel expenses. 
                
                Your final answer must be a detailed
                report on the chosen city, and everything you found out
                about it, including the actual flight costs, weather 
                forecast and attractions.
                {self.__tip_section()}

                Traveling from: {origin}
                City Options: {cities}
                Trip Date: {range}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Detailed report on the chosen city including flight costs, weather forecast, and attractions"
        )

    def gather_task(self, agent, origin, interests, range, extra_context=""):
        return Task(
            description=dedent(f"""
                {extra_context}
                As a local expert on this city you must compile an 
                in-depth guide for someone traveling there and wanting 
                to have THE BEST trip ever!
                Gather information about key attractions, local customs,
                special events, and daily activity recommendations.
                Find the best spots to go to, the kind of place only a
                local would know.
                This guide should provide a thorough overview of what 
                the city has to offer, including hidden gems, cultural
                hotspots, must-visit landmarks, weather forecasts, and
                high level costs.
                
                The final answer must be a comprehensive city guide, 
                rich in cultural insights and practical tips, 
                tailored to enhance the travel experience.
                {self.__tip_section()}

                Trip Date: {range}
                Traveling from: {origin}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Comprehensive city guide including hidden gems, cultural hotspots, and practical travel tips"
        )

    @staticmethod
    def _infer_num_days(date_range: str) -> int:
        """Infer the number of days from a date-range string like 'June 1-7, 2026'."""
        m = re.search(r'(\d{1,2})\s*[-–]\s*(\d{1,2})', date_range)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if end > start:
                return end - start + 1
        m = re.search(r'(\d+)\s*[-]?\s*days?', date_range, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 7  # fallback

    def plan_task(self, agent, origin, interests, range, extra_context=""):
        num_days = self._infer_num_days(range)
        day_headers = ", ".join(f"Day {i}" for i in builtins.range(1, num_days + 1))
        return Task(
            description=dedent(f"""
                {extra_context}
                Expand this guide into a full {num_days}-day travel
                itinerary with detailed per-day plans, including
                weather forecasts, places to eat, packing suggestions,
                and a budget breakdown.

                You MUST suggest actual places to visit, actual hotels
                to stay and actual restaurants to go to.

                This itinerary should cover all aspects of the trip,
                from arrival to departure, integrating the city guide
                information with practical travel logistics.

                IMPORTANT FORMATTING RULES:
                - Structure your itinerary with EXACTLY {num_days} days.
                - Use explicit day headers: {day_headers}
                - Each day MUST start with a markdown heading like:
                  ## Day 1: [Date / Title]
                  ## Day 2: [Date / Title]
                  ... and so on up to Day {num_days}.
                - Under each day, list Morning, Afternoon, and Evening
                  activities with specific venues and times.
                - Do NOT repeat the same venue or restaurant on
                  multiple days.
                - End with a Budget Breakdown section showing
                  estimated total cost.

                Your final answer MUST be a complete expanded travel plan,
                formatted as markdown, encompassing a daily schedule,
                anticipated weather conditions, recommended clothing and
                items to pack, and a detailed budget, ensuring THE BEST
                TRIP EVER. Be specific and give it a reason why you picked
                each place, what makes them special! {self.__tip_section()}

                Trip Date: {range}
                Traveling from: {origin}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output=f"Complete {num_days}-day travel plan with Day 1 through Day {num_days} headers, daily schedule, weather conditions, packing suggestions, and budget breakdown"
        )

    def __tip_section(self):
        return "If you do your BEST WORK, I'll tip you $100!"
