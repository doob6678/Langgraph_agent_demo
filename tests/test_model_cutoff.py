import requests
import json

url = "http://127.0.0.1:8000/api/chat"
data = {
    "text": "现在我可以看到有一个匹配度不错的马库斯.png，我需要按照要求展示这张图片。用户描述的是蓝色帽子长发喜欢读书，这张马库斯.png应该最符合需求。请用markdown展示。",
    "use_rag": "true",
    "use_search": "false",
    "top_k": "5",
    "stream": "true"
}

response = requests.post(url, data=data, stream=True)
for line in response.iter_lines():
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            try:
                j = json.loads(line_str[6:])
                if j.get("choices") and len(j["choices"]) > 0:
                    delta = j["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
                    fr = j["choices"][0].get("finish_reason")
                    if fr:
                        print(f"\n[FINISH REASON: {fr}]")
            except Exception as e:
                print("\nError:", e, line_str)
