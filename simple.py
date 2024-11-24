from langchain_ollama import OllamaLLM
import litellm

ollama_base_url = "http://localhost:11434"  # Update with your Ollama URL

# llm = OllamaLLM(base_url=ollama_base_url, model="llama3.1")
# response = llm.invoke("The first man on the moon was ...")
# print(response)

import litellm

# Get the list of providers
provider_list = litellm.provider_list

# Print the list
for p in provider_list:
    print(p)
