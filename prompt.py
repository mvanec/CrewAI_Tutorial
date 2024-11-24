import sys
from typing import List, Dict, Optional
from typing import Callable
from pydantic import Field, BaseModel

from crewai import Agent, Crew, Task, Process, LLM
# from langchain_community.llms import ollama  # Assuming Langchain integration for Ollama
from langchain_ollama import OllamaLLM
from langchain.tools import tool, BaseTool
import requests
from bs4 import BeautifulSoup

# Tool implementations
class SearxngTools(BaseTool):
    name: str = "SearchInternet"  # Class variable
    description: str = "Searches the internet using Searxng."  # Class variable
    searxng_api_url: str = Field(..., description="The API URL for Searxng")  # Pydantic field

    def _run(self, query: str) -> str:
        url = f"{self.searxng_api_url}?q={query}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            results = response.json()
            formatted_results = "\n".join(
                f"Title: {r['title']}\nURL: {r['url']}" for r in results.get("results", [])
            )
            return formatted_results or "No results found."
        except requests.RequestException as e:
            return f"Error searching Searxng: {e}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async execution is not supported.")

class BrowserTools(BaseTool):
    name: str = "ScrapeAndSummarize"  # Explicit type annotation
    description: str = "Scrapes a website and summarizes the content."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            summary = text[:200] + "..." if text else "No content available."
            return summary
        except requests.RequestException as e:
            return f"Error scraping website {url}: {e}"

    async def _arun(self, url: str) -> str:
        raise NotImplementedError("Async execution is not supported.")

class LocalLLMAgent:
    def __init__(self, ollama_base_url: str, searxng_api_url: str):
        self.ollama_llm = OllamaLLM(base_url=ollama_base_url, model="llama3.1")
        self.searxng_api_url = searxng_api_url

    def run(self, user_input: str):
        task_planner = self.create_task_planner_agent()
        web_researcher = self.create_web_researcher_agent()
        initial_task = self.create_initial_task(user_input)

        crew = Crew(
            agents=[task_planner, web_researcher],
            tasks=[initial_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return result

    def create_task_planner_agent(self) -> Agent:
        print("******************** Before agent", file=sys.stderr)

        agent = Agent(
            name="TaskPlanner",
            role="Decomposes user requests into actionable sub-tasks",
            goal="Determine the necessary steps and tools to fulfill the user's request.",
            backstory="You are a task planner",
            llm=self.ollama_llm,
            verbose=True,
        )
        print(f"******************** After agent", file=sys.stderr)

        return agent


    def create_web_researcher_agent(self) -> Agent:

        search_tool = SearxngTools(searxng_api_url=self.searxng_api_url)  # Pass URL during instantiation
        scrape_tool = BrowserTools()  # No extra fields, can be instantiated directly

        print(f"****************** The URL is {search_tool.searxng_api_url}", file=sys.stderr)

        return Agent(
            name="WebResearcher",
            role="Searches the web and gathers information.",
            goal="Retrieve and summarize relevant information from search results and websites.",
            llm=self.ollama_llm,
            backstory="You are a researcher who searches the Web to respond to client requests.",
            tools=[search_tool, scrape_tool],
            verbose=True,
            tool_code_interpreter="python"
        )

    def create_initial_task(self, user_input: str) -> Task:
         return Task(
            description=f"Analyze the user's request and create a plan to address it. "
                        f"User request: {user_input}",
            agent="TaskPlanner"  # Use agent name as string
         )


class ToolValidator(BaseModel):
    name: str
    description: str
    func: Callable

# Validate an instance of SearxngTools
try:
    search_tool_instance = SearxngTools(searxng_api_url="https://searx.org/api")
    tool_data = ToolValidator(
        name=search_tool_instance.name,
        description=search_tool_instance.description,
        func=search_tool_instance._run  # Pass the instance method
    )
    print("SearxngTools validation passed!")
except Exception as e:
    print(f"SearxngTools validation error: {e}")

# Validate an instance of BrowserTools
try:
    scrape_tool_instance = BrowserTools()
    tool_data = ToolValidator(
        name=scrape_tool_instance.name,
        description=scrape_tool_instance.description,
        func=scrape_tool_instance._run  # Pass the instance method
    )
    print("BrowserTools validation passed!")
except Exception as e:
    print(f"BrowserTools validation error: {e}")

# Example Usage
ollama_base_url = "http://localhost:11434"  # Update with your Ollama URL
searxng_api_url = "https://searx.org/api" # Public searxng instance; replace if you have your own

# agent = LocalLLMAgent(ollama_base_url, searxng_api_url)
# user_input = "What are the current top 3 news headlines about AI?"
# result = agent.run(user_input)
# print(result)
