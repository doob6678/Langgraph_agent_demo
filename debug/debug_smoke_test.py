import requests
import json
import sys
import time

def test_api():
    # 默认假设后端运行在 8000 端口，如果不是请修改
    url = "http://127.0.0.1:8000/api/chat"
    
    # 模拟前端发送的表单数据
    payload = {
        "text": "Hello, smoke test",
        "use_rag": "false",
        "use_search": "false",
        "top_k": "3",
        "stream": "true"
    }
    
    print(f"[*] Connecting to {url}...")
    print(f"[*] Payload: {payload}")
    
    try:
        t0 = time.time()
        # 设置 stream=True 以获取流式响应
        with requests.post(url, data=payload, stream=True, timeout=30) as response:
            print(f"[*] Status Code: {response.status_code}")
            print(f"[*] Headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"[!] Error Body: {response.text}")
                return

            print("\n--- Stream Content Start ---")
            line_count = 0
            has_content = False
            
            for line in response.iter_lines():
                # filter out keep-alive new lines
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"RAW: {decoded_line}")
                    line_count += 1
                    
                    if decoded_line.startswith("data:"):
                        has_content = True
                        data_content = decoded_line[5:].strip()
                        if data_content == "[DONE]":
                            print("    -> Received [DONE] signal")
                        else:
                            try:
                                json_data = json.loads(data_content)
                                # 尝试提取 delta content
                                if "choices" in json_data:
                                    delta = json_data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        print(f"    -> Content Chunk: {repr(content)}")
                            except:
                                pass
            
            print("--- Stream Content End ---")
            print(f"[*] Total Time: {time.time() - t0:.2f}s")
            
            if line_count == 0:
                print("[!] CRITICAL: Received 200 OK but body was completely empty.")
            elif not has_content:
                print("[!] CRITICAL: Received data but no 'data:' prefix lines found (Not SSE format).")
            else:
                print("[*] Smoke test passed: Received valid SSE stream.")

    except requests.exceptions.ConnectionError:
        print(f"[!] Connection refused. Please ensure the backend is running on port 8000.")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")

if __name__ == "__main__":
    test_api()
