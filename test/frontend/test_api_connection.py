
import sys
import os
import requests
import time
import json

# 配置
BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

def test_chat_api():
    print("=== 开始测试前端 Chat API ===")
    
    # 1. 检查服务是否健康 (简单 GET)
    try:
        resp = requests.get(BASE_URL)
        if resp.status_code == 200:
            print("[PASS] 服务主页访问正常")
        else:
            print(f"[FAIL] 服务主页访问失败: {resp.status_code}")
            return
    except Exception as e:
        print(f"[FAIL] 服务连接失败: {e}")
        return

    # 2. 发送聊天请求
    payload = {
        "text": "你好，请记住我叫测试员007",
        "use_rag": True,
        "use_search": False,
        "top_k": 5,
        "stream": False  # 先测试非流式
    }
    
    print(f"\n[INFO] 发送聊天请求: {payload['text']}")
    start_time = time.time()
    
    try:
        resp = requests.post(CHAT_ENDPOINT, data=payload)
        end_time = time.time()
        duration = end_time - start_time
        
        if resp.status_code == 200:
            print(f"[PASS] 请求成功，耗时: {duration:.2f}秒")
            data = resp.json()
            print(f"[INFO] 响应内容: {json.dumps(data, ensure_ascii=False)[:200]}...")
            
            # 检查是否有回复
            if data.get("response"):
                print("[PASS] 收到有效回复")
            else:
                print("[FAIL] 回复内容为空")
                
        else:
            print(f"[FAIL] 请求失败: {resp.status_code}")
            print(f"错误详情: {resp.text}")
            
    except Exception as e:
        print(f"[FAIL] 请求异常: {e}")

def test_memory_recall():
    print("\n=== 测试记忆召回 ===")
    payload = {
        "text": "我是谁？",
        "use_rag": True,
        "use_search": False,
        "stream": False
        # 注意：这里 user_id 是匿名的，如果没有持久化 cookie/session，可能记不住
        # 但在同一脚本中，如果后端是基于 user_id 字段... 我们这里没传 user_id，后端会生成新的
    }
    
    # 为了测试记忆，我们需要手动传递 user_id
    # 但由于上一次请求没返回 user_id (除非后端改了)，我们这里只能模拟一个新的带 ID 的请求流程
    
    # 重新来一轮带 user_id 的
    user_id = "test_user_123"
    payload1 = {
        "text": "记住我的代号是 Alpha",
        "user_id": user_id,
        "stream": False
    }
    print(f"[INFO] 存储记忆请求: {payload1['text']}")
    requests.post(CHAT_ENDPOINT, data=payload1)
    
    payload2 = {
        "text": "我的代号是什么？",
        "user_id": user_id,
        "stream": False
    }
    print(f"[INFO] 召回记忆请求: {payload2['text']}")
    resp = requests.post(CHAT_ENDPOINT, data=payload2)
    if resp.status_code == 200:
        ans = resp.json().get("response", "")
        print(f"[INFO] 回复: {ans}")
        if "Alpha" in ans or "alpha" in ans:
            print("[PASS] 记忆召回成功")
        else:
            print("[WARN] 记忆召回可能失败，需人工核对")

if __name__ == "__main__":
    test_chat_api()
    test_memory_recall()
