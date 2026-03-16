import requests
import json
import sys

def test_stream():
    url = "http://127.0.0.1:8000/api/chat"
    data = {
        "text": "我也想要个蓝色的帽子",
        "use_rag": "true",
        "use_search": "false",
        "stream": "true"
    }
    
    print(f"Sending request to {url}...")
    try:
        with requests.post(url, data=data, stream=True) as r:
            if r.status_code != 200:
                print(f"Error: {r.status_code}")
                print(r.text)
                return

            print("Receiving stream...")
            for line in r.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]
                        if json_str == '[DONE]':
                            print("\n[DONE]")
                            break
                        try:
                            chunk = json.loads(json_str)
                            # Check for tool events
                            if 'x_tool_event' in chunk:
                                print(f"Tool Event: {chunk['x_tool_event'].get('tool')}")
                                continue
                            
                            choices = chunk.get('choices', [])
                            if not choices:
                                continue
                                
                            content = choices[0].get('delta', {}).get('content', '')
                            if content:
                                print(f"Chunk: {repr(content)}")
                                # Check for split markdown tags
                                if content.endswith('![') or content.endswith('!'):
                                    print("WARNING: Chunk ends with potential split markdown tag!")
                        except json.JSONDecodeError:
                            print(f"JSON Error: {json_str}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_stream()
