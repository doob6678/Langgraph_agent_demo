import requests
import json

url = "http://127.0.0.1:8000/api/chat"
data = {
    "text": "蓝色",
    "use_rag": "true",
    "use_search": "false",
    "top_k": "5",
    "stream": "true"
}

print("Sending request...")
response = requests.post(url, data=data, stream=True)
print("Response status:", response.status_code)

for line in response.iter_lines():
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            if line_str.strip() == "data: [DONE]":
                print("\n[STREAM DONE]")
                continue
            try:
                j = json.loads(line_str[6:])
                if "x_tool_event" in j:
                    print(f"\n[TOOL] {j['x_tool_event']['tool']} - {j['x_tool_event']['status']}")
                    if j['x_tool_event']['status'] == 'completed':
                        print(f"[TOOL RESULT] {j['x_tool_event'].get('result_preview', '')}")
                elif j.get("choices") and len(j["choices"]) > 0:
                    delta = j["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
                    fr = j["choices"][0].get("finish_reason")
                    if fr:
                        print(f"\n[FINISH REASON: {fr}]")
            except Exception as e:
                print("\nError parsing JSON:", e, line_str)
