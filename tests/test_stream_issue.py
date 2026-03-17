import requests

url = "http://127.0.0.1:8000/api/chat"
data = {
    "text": "请展示马库斯.png，记得用markdown格式",
    "use_rag": "true",
    "use_search": "false",
    "top_k": "5",
    "stream": "true"
}

response = requests.post(url, data=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
