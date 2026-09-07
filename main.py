from crewai import Crew
from trip_agents import TripAgents
from trip_tasks import TripTasks
from reflexion_memory import ReflexionMemory
from evaluator import evaluate

from dotenv import load_dotenv
load_dotenv()


class TripCrew:

    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        
        memory = ReflexionMemory()
        task_description = (
            f"Plan a trip from {self.origin} to one of {self.cities} "
            f"during {self.date_range}, interests: {self.interests}"
        )

        past_reflections = memory.retrieve_relevant(task_description)
        reflexion_context = memory.build_reflexion_context(task_description)
    

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range,
            extra_context=reflexion_context,
        )
        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range,
            extra_context=reflexion_context,
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.origin,
            self.interests,
            self.date_range,
            extra_context=reflexion_context,
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        output_text = str(result)

    
        task_input = {
            "origin": self.origin,
            "cities": self.cities,
            "date_range": self.date_range,
            "interests": self.interests,
        }
        eval_result = evaluate(task_input, output_text)
        print(f"\n📊 Accuracy: {eval_result['accuracy']:.2f} "
              f"({eval_result['checks_passed']}/{eval_result['checks_applicable']} checks)")


        memory.reflect_and_store(task_description, output_text, eval_result)
        

        return result


if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')

    location = "Mumbai"
    cities = "Paris, Rome, Barcelona"
    date_range = "June 1-7, 2026"
    interests = "art, food, history"

    print(f"From where will you be traveling from?\n{location}\n")
    print(f"What are the cities options you are interested in visiting?\n{cities}\n")
    print(f"What is the date range you are interested in traveling?\n{date_range}\n")
    print(f"What are some of your high level interests and hobbies?\n{interests}\n")

    trip_crew = TripCrew(location, cities, date_range, interests)
    result = trip_crew.run()

    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)