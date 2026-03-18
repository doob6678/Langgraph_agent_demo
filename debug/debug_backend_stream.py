
import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加路径
sys.path.append(os.getcwd())

load_dotenv()

from backend.agent.graph_new_real import agent_graph
from backend.agent.state_ex.agent_state import AgentState

async def debug_stream():
    print("=== 开始调试后端事件流 ===")
    
    state = AgentState(
        messages=[],
        user_input="你好，我是调试员",
        user_id="debug_user_001",
        tool_flags=[True, True],
        top_k=5
    )
    
    print(f"输入状态: {state}")
    
    try:
        # 使用 astream_events 打印所有事件
        async for event in agent_graph.astream_events(state, version="v1"):
            kind = event["event"]
            name = event.get("name")
            
            print(f"\n[Event] Kind: {kind}, Name: {name}")
            
            if kind == "on_chain_end":
                data = event.get("data", {})
                output = data.get("output")
                print(f"  -> Output keys: {output.keys() if isinstance(output, dict) else type(output)}")
                if name == "memory_recall":
                    print(f"  -> !!! FOUND memory_recall END !!!")
                    print(f"  -> Output: {output}")
                    
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(debug_stream())
