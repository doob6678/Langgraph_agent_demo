import asyncio
import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agent.tool_ex.tools import analyze_image, rag_image_search
from backend.services.image_service import image_service

def test_image_search():
    image_path = r"D:\project2026\rag_agent_app\research\memory_manage\milvus_demo\image_search\img_to_search\槲寄生小姐.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    import base64
    b64_data = base64.b64encode(image_bytes).decode('utf-8')

    print("--- Running analyze_image ---")
    analyze_result = analyze_image.invoke({"image_data_base64": b64_data, "description": "这张图能找到一样的rag吗"})
    print("Analyze Result:", analyze_result)

    print("--- Running rag_image_search ---")
    rag_result = rag_image_search.invoke({"query": "槲寄生小姐"})
    print("Rag Result:", rag_result)

if __name__ == "__main__":
    test_image_search()
