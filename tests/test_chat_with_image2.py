import requests

url = "http://127.0.0.1:8000/api/chat_with_image"
image_path = r"D:\project2026\rag_agent_app\research\memory_manage\milvus_demo\image_search\img_to_search\槲寄生小姐.png"

with open(image_path, "rb") as f:
    files = {"image": ("槲寄生小姐.png", f, "image/png")}
    data = {
        "message": "这张图能找到一样的rag吗",
        "use_rag": "true",
        "use_search": "true",
        "top_k": "5",
        "stream": "true"
    }
    response = requests.post(url, data=data, files=files, stream=True)
    
    full_text = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                import json
                try:
                    data_json = json.loads(data_str)
                    if "choices" in data_json and len(data_json["choices"]) > 0:
                        content = data_json["choices"][0].get("delta", {}).get("content", "")
                        full_text += content
                except Exception as e:
                    pass
    print("FINAL LLM OUTPUT:")
    print(full_text)
