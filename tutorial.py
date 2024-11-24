from crewai import Crew, Agent, Task
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool
from crewai.tools import BaseTool

# Website Data Ingestion
loader = UnstructuredURLLoader(
    # urls=["https://docs.crewai.com/how-to/Installing-CrewAI/"]
    urls=["https://www.mandolessons.com/lessons/fiddle-tunes/the-butterfly/"],
    show_progress_bar=True
)

data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

# Initialize Embeddings
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# print(f"Embeddings = {embeddings}")
# quit()

# Create and Persist Vector Database
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
db.persist()

# Define Retriever from Vector Store
retriever = db.as_retriever()

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "Result from custom tool"

researcher = Agent(
    ai=model_name,
    name="Researcher",
    role="Researches topics by searching the website data.",
    # tools=[
    #     Tool(
    #         name="Website_Search",
    #         func=retriever.get_relevant_documents,
    #         description="useful for when you need to ask with lookup on website data.",
    #     )
    # ],
    tools=[MyCustomTool()],
    goal="Answer questions by retrieving relevant information from the website's data.",  # Add the goal here
    backstory="You are a helpful AI assistant specializing in searching and retrieving information from a website. Use your 'Website_Search' tool to find relevant documents when answering questions.",
)

researcher_boss = Agent(
    ai=model_name,
    name="Researcher Boss",
    role="Challenges the researcher to bring out the best out of his findings",
    tools=[
        {
            "name": "Website_Search",
            "func": retriever.get_relevant_documents,
            "description": "useful for validating the findings of the researcher",
        }
    ],
    # tools=[MyCustomTool()],
    goal="Ask further questions to the researcher and validates the retrieved relevant information from the website's data.",  # Add the goal here
    backstory="You are a helpful AI assistant boss and your job is to make sure the retrieved information is correct. Use your 'Website_Search' tool to find relevant documents when answering questions.",
)

research_task = Task (
    description=(
        "Analyze the URL provided ({crewai_url}) "
        "to extract information about how crewai works. "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of crewai specifications, including necessary "
        "tools to get started"
    ),
    agent=researcher,
    async_execution=True,
)

boss_task = Task (
    description=(
        "Analyze the URL provided ({crewai_url}) "
        "to extract information about how crewai works. "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of crewai specifications, including necessary "
        "tools to get started"
    ),
    agent=researcher_boss,
    async_execution=True,
)

# Create Crew
research_crew1 = Crew(
    agents=[researcher, researcher_boss],
    tasks=[research_task, boss_task],
    verbose=True,  # This will print logs to the console as the crew works
)

# Job Context
job_crew_works = {
    "crewai_url": "https://docs.crewai.com/how-to/Installing-CrewAI/",
    "personal_writeup": """Accomplished Researcher
    with 18 years of experience, specializing in
    setting up CrewAI kind of agent based systems""",
}

# Kickoff the Crew's Work
result = research_crew1.kickoff(inputs=job_crew_works)
print(result)
