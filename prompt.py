import os
from typing import List, Dict

from crewai import Agent, Crew, Task, Process
from langchain_community.llms import ollama  # Assuming Langchain integration for Ollama
from langchain_ollama import OllamaLLM
from langchain.tools import tool, Tool
import requests
from bs4 import BeautifulSoup

# Tool implementations
class SearxngTools:
    @staticmethod
    @tool
    def search_internet(query: str, searxng_api_url: str) -> str:
        """Searches the internet using Searxng and returns the results."""
        url = f"{searxng_api_url}?q={query}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            results = response.json()
            # Process Searxng results to a user-friendly format
            formatted_results = ""
            for result in results["results"]:
                formatted_results += f"Title: {result['title']}\nURL: {result['url']}\n\n"
            return formatted_results
        except requests.exceptions.RequestException as e:
            return f"Error searching Searxng: {e}"



class BrowserTools:
    @staticmethod
    @tool
    def scrape_and_summarize_website(url: str) -> str:
        """Scrapes a website and returns a summary."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()  # Extract all text from the page

            # Basic summarization (you might want to use a dedicated summarization library)
            # Example using the first 200 characters as a very basic summary
            summary = text[:200] + "..."
            return summary
        except requests.exceptions.RequestException as e:
            return f"Error scraping website {url}: {e}"


class LocalLLMAgent:
    def __init__(self, ollama_base_url: str, searxng_api_url: str):
        self.ollama_llm = OllamaLLM(base_url=ollama_base_url, model="llama2")
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
        return Agent(
            name="TaskPlanner",
            role="Decomposes user requests into actionable sub-tasks",
            goal="Determine the necessary steps and tools to fulfill the user's request.",
            backstory="You are a task planner",
            llm=self.ollama_llm,
            verbose=True,
        )


    def create_web_researcher_agent(self) -> Agent:

        search_tool = {
            "name": "SearchInternet",
            "func": SearxngTools.search_internet,
            "description": "Searches the internet using Searxng."
        }
        scrape_tool = {
            "name": "ScrapeAndSummarize",
            "func": BrowserTools.scrape_and_summarize_website,
            "description": "Scrapes a website and summarizes the content."
        }

        return Agent(
            name="WebResearcher",
            role="Searches the web and gathers information.",
            goal="Retrieve and summarize relevant information from search results and websites.",
            llm=self.ollama_llm,
            backstory="You are a researcher who searches the Web to respond to client requests.",
            tools=[search_tool, scrape_tool],
            verbose=True,
            tool_code_interpreter="python"  # Crucial for passing parameters
        )


    def create_initial_task(self, user_input: str) -> Task:
         return Task(
            description=f"Analyze the user's request and create a plan to address it. "
                        f"User request: {user_input}",
            agent="TaskPlanner"  # Use agent name as string
         )



# Example Usage
ollama_base_url = "http://localhost:11434"  # Update with your Ollama URL
searxng_api_url = "https://searx.org/api" # Public searxng instance; replace if you have your own

agent = LocalLLMAgent(ollama_base_url, searxng_api_url)
user_input = "What are the current top 3 news headlines about AI?"
result = agent.run(user_input)
print(result)