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
    
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
