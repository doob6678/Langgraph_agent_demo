import requests
import json
import sys

def test_stream_chat():
    url = "http://127.0.0.1:8000/api/chat"
    data = {
        "text": "你好，请介绍一下你自己",
        "use_rag": "false",
        "use_search": "false",
        "stream": "true"
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, data=data, stream=True, timeout=30)
        response.raise_for_status()
        
        print("Response status code:", response.status_code)
        
        received_content = ""
        received_chunks = 0
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        print("\n[DONE] received")
                        break
                    
                    try:
                        data_json = json.loads(data_str)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            delta = data_json["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                received_content += content
                                received_chunks += 1
                    except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {data_str}")
        
        print(f"\n\nTest completed. Received {received_chunks} chunks.")
        if received_chunks > 0:
            print("SUCCESS: Stream received correctly.")
        else:
            print("FAILURE: No content received.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_stream_chat()
