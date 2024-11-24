from langchain_ollama import OllamaLLM
ollama_base_url = "http://localhost:11434"  # Update with your Ollama URL

llm = OllamaLLM(base_url=ollama_base_url, model="llama3.1")
response = llm.invoke("The first man on the moon was ...")
print(response)
