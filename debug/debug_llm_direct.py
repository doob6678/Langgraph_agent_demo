import os
import sys
import asyncio
import time

# Ensure backend is in path
sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.agent.util_ex.common import get_langchain_chat_model

load_dotenv()

async def test_llm():
    print("[-] Testing LLM connection directly...")
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL")
    print(f"[-] API Key present: {bool(api_key)}")
    print(f"[-] Base URL: {base_url}")
    
    llm = get_langchain_chat_model(
        model="doubao-seed-2-0-lite-260215",
        temperature=0.7,
        max_tokens=100,
        streaming=True
    )
    
    msg = HumanMessage(content="Hello, are you there?")
    print("[-] Invoking LLM (ainvoke)... please wait")
    
    try:
        t0 = time.time()
        # We use ainvoke which is what the node uses
        res = await llm.ainvoke([msg])
        t1 = time.time()
        print(f"[+] Response received in {t1-t0:.2f}s")
        print(f"[+] Content: {res.content}")
    except Exception as e:
        print(f"[!] Error invoking LLM: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_llm())
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
