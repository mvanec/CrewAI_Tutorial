import requests

from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.mandolessons.com/lessons/fiddle-tunes/the-butterfly/",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]

url = 'https://www.mandolessons.com/lessons/fiddle-tunes/the-butterfly/'
response = requests.get(url)
print(response.status_code)

headers = {"User-Agent": "Mozilla/5.0"}
loader = UnstructuredURLLoader(urls=urls, ssl_verify=True, headers=headers)

data = loader.load()

print(data[0])