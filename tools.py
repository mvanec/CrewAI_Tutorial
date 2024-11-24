from typing import override
from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool

ollama_url = "http://localhost:11434"

class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "Performs basic mathematical calculations"

    def _run(self, expression: str):
        try:
            result = eval(expression)
            return result
        except Exception as e:
            return "Error: {}".format(e)


class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Searches the internet for information"

    def _run(self, query: str):
        # Implement the search functionality here
        return "Search results for {}".format(query)


tools = {"search": SearchTool(), "calculator": Calculator()}

# Initialize OllamaLLM with your local instance URL
crew_llm = LLM (
    model="ollama/llama3.1",
    base_url=ollama_url
)

agent = Agent(
    role="calculator",
    goal="Perform a mathematical calculation",
    backstory="I've been created to solve exciting math problems from the siimplest to the most commplex.",
    tools=[SearchTool(), Calculator()],
    llm=crew_llm
)

# Now you can use the agent with its tools
# Create a task for the agent
task = Task(
    description="Solve a math problem",
    agent=agent,
    input="Calculate 2 + 2",
    expected_output="The result of the calculation as a number",  # Specify the expected output
)

# Create a crew with one or more agents
crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,  # Or Process.parallel
    verbose=True,
)

# Run the crew
result = crew.kickoff()
print(f"Result: {result}")
